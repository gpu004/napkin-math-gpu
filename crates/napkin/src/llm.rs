//! First-principles memory and operation counts for decoder-only transformer serving.

use std::fmt;
use std::num::NonZeroU64;

use crate::profile::Dtype;

/// Transformer dimensions and precision assumptions used by a serving recipe.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    /// Number of transformer layers, `L`.
    pub layers: NonZeroU64,
    /// Hidden width, `d`, used for the activation floor.
    pub hidden: NonZeroU64,
    /// Query heads, `H`; attention FLOPs use query heads even with GQA.
    pub heads: NonZeroU64,
    /// Key/value heads, `Hkv`; KV-cache bytes scale with this count.
    pub kv_heads: NonZeroU64,
    /// Head width, `dh`.
    pub head_dim: NonZeroU64,
    /// Total model parameters; all are assumed to be matmul weights.
    pub params: f64,
    /// Weight storage precision in bits per parameter.
    pub weight_bits: f64,
    /// KV-cache storage precision in bits per element.
    pub kv_bits: f64,
}

/// Batch and sequence lengths for a decoder-only serving request.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// Number of concurrent requests, `B`.
    pub batch: NonZeroU64,
    /// Prompt length per request, `S`.
    pub prompt_tokens: NonZeroU64,
    /// Generated length per request, `G`.
    pub generated_tokens: NonZeroU64,
    /// Compute and activation dtype; activations use the dtype's storage size.
    pub activation: Dtype,
}

/// FLOPs and HBM traffic for one phase.
#[derive(Clone, Copy, Debug)]
pub struct Phase {
    /// Tokens processed by this phase (`B × S` for prefill, `B` for decode).
    pub tokens: f64,
    /// Linear-layer operations, `2 × params × tokens`.
    pub linear_flops: f64,
    /// Attention operations; prefill skips the masked causal half, decode uses
    /// all `c` context keys. Both count query heads, not KV heads.
    pub attention_flops: f64,
    /// Model weights read once for this phase, shared across the batch.
    pub weight_bytes: f64,
    /// KV bytes written during prefill or read during a decode step.
    pub kv_bytes: f64,
    /// Activation floor: one input read and output write per layer and token.
    pub activation_bytes: f64,
}

impl Phase {
    /// Total linear plus attention FLOPs.
    #[must_use]
    pub fn flops(self) -> f64 {
        self.linear_flops + self.attention_flops
    }

    /// Total weight, KV-cache, and activation HBM bytes.
    #[must_use]
    pub fn hbm_bytes(self) -> f64 {
        self.weight_bytes + self.kv_bytes + self.activation_bytes
    }
}

/// Derived model-resident bytes and one prefill/decode serving step.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// `params × weight_bits / 8`, with all parameters treated as weights.
    pub weight_bytes: f64,
    /// `2 × layers × KV heads × head_dim × kv_bits / 8`.
    pub kv_bytes_per_token: f64,
    /// `batch × (prompt_tokens + generated_tokens)` across all requests.
    pub live_tokens: f64,
    /// `kv_bytes_per_token × live_tokens`; resident capacity, not per-step traffic.
    pub kv_cache_bytes: f64,
    /// Prefill processes all prompts in one forward pass; causal attention
    /// counts only the unmasked half of the sequence square.
    pub prefill: Phase,
    /// Decode models the final, largest step at context `prompt + generated`;
    /// one newly written KV row is ignored.
    pub decode: Phase,
    /// `prompt_tokens + generated_tokens`, the context used for the last decode step.
    pub decode_context: f64,
}

/// A recipe that could not be derived.
#[derive(Debug)]
pub struct LlmError(pub String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LlmError {}

fn as_f64(value: NonZeroU64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "dimensions and token counts are approximate arithmetic inputs"
    )]
    let converted = value.get() as f64;
    converted
}

/// Storage bytes per element; tf32 is stored as fp32.
#[must_use]
pub fn element_bytes(dtype: Dtype) -> f64 {
    match dtype {
        Dtype::Fp4 => 0.5,
        Dtype::Fp8 => 1.0,
        Dtype::Fp16 | Dtype::Bf16 => 2.0,
        Dtype::Tf32 | Dtype::Fp32 => 4.0,
    }
}

fn finite_recipe(recipe: &Recipe) -> bool {
    [
        recipe.weight_bytes,
        recipe.kv_bytes_per_token,
        recipe.live_tokens,
        recipe.kv_cache_bytes,
        recipe.prefill.tokens,
        recipe.prefill.linear_flops,
        recipe.prefill.attention_flops,
        recipe.prefill.weight_bytes,
        recipe.prefill.kv_bytes,
        recipe.prefill.activation_bytes,
        recipe.decode.tokens,
        recipe.decode.linear_flops,
        recipe.decode.attention_flops,
        recipe.decode.weight_bytes,
        recipe.decode.kv_bytes,
        recipe.decode.activation_bytes,
        recipe.decode_context,
    ]
    .into_iter()
    .all(f64::is_finite)
}

/// Derives model memory and prefill/decode FLOPs and HBM traffic.
///
/// # Errors
/// Returns `LlmError` when model precisions or parameter count are not positive
/// and finite, the query/KV head ratio is not integral, or derived values overflow.
pub fn recipe(model: &Model, workload: &Workload) -> Result<Recipe, LlmError> {
    if !(model.params > 0.0 && model.params.is_finite()) {
        return Err(LlmError("params must be positive and finite".to_owned()));
    }
    if !(model.weight_bits > 0.0 && model.weight_bits.is_finite()) {
        return Err(LlmError(
            "weight bits must be positive and finite".to_owned(),
        ));
    }
    if !(model.kv_bits > 0.0 && model.kv_bits.is_finite()) {
        return Err(LlmError("KV bits must be positive and finite".to_owned()));
    }
    if !model.heads.get().is_multiple_of(model.kv_heads.get()) {
        return Err(LlmError(
            "heads must be divisible by KV heads for integral GQA groups".to_owned(),
        ));
    }

    let layers = as_f64(model.layers);
    let hidden = as_f64(model.hidden);
    let heads = as_f64(model.heads);
    let kv_heads = as_f64(model.kv_heads);
    let head_dim = as_f64(model.head_dim);
    let batch = as_f64(workload.batch);
    let prompt = as_f64(workload.prompt_tokens);
    let generated = as_f64(workload.generated_tokens);
    let activation = element_bytes(workload.activation);
    let decode_context = prompt + generated;
    let weight_bytes = model.params * model.weight_bits / 8.0;
    let kv_bytes_per_token = 2.0 * layers * kv_heads * head_dim * model.kv_bits / 8.0;
    let live_tokens = batch * decode_context;
    let kv_cache_bytes = kv_bytes_per_token * live_tokens;
    let prefill_tokens = batch * prompt;
    let prefill = Phase {
        tokens: prefill_tokens,
        linear_flops: 2.0 * model.params * prefill_tokens,
        attention_flops: 2.0 * layers * heads * head_dim * batch * prompt * prompt,
        weight_bytes,
        kv_bytes: kv_bytes_per_token * prefill_tokens,
        activation_bytes: 2.0 * layers * batch * prompt * hidden * activation,
    };
    let decode = Phase {
        tokens: batch,
        linear_flops: 2.0 * model.params * batch,
        attention_flops: 4.0 * layers * heads * head_dim * batch * decode_context,
        weight_bytes,
        kv_bytes: kv_bytes_per_token * batch * decode_context,
        activation_bytes: 2.0 * layers * batch * hidden * activation,
    };
    let recipe = Recipe {
        weight_bytes,
        kv_bytes_per_token,
        live_tokens,
        kv_cache_bytes,
        prefill,
        decode,
        decode_context,
    };
    if !finite_recipe(&recipe) {
        return Err(LlmError(
            "derived recipe quantities must be finite".to_owned(),
        ));
    }
    Ok(recipe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Model, Workload) {
        (
            Model {
                layers: NonZeroU64::new(2).unwrap(),
                hidden: NonZeroU64::new(64).unwrap(),
                heads: NonZeroU64::new(8).unwrap(),
                kv_heads: NonZeroU64::new(2).unwrap(),
                head_dim: NonZeroU64::new(8).unwrap(),
                params: 1e6,
                weight_bits: 8.0,
                kv_bits: 16.0,
            },
            Workload {
                batch: NonZeroU64::new(3).unwrap(),
                prompt_tokens: NonZeroU64::new(10).unwrap(),
                generated_tokens: NonZeroU64::new(6).unwrap(),
                activation: Dtype::Bf16,
            },
        )
    }

    #[test]
    fn derives_literal_prefill_decode_and_cache_values() {
        let (model, workload) = fixture();
        let recipe = recipe(&model, &workload).unwrap();
        assert_eq!(recipe.kv_bytes_per_token, 128.0);
        assert_eq!(recipe.weight_bytes, 1e6);
        assert_eq!(recipe.live_tokens, 48.0);
        assert_eq!(recipe.kv_cache_bytes, 6144.0);
        assert_eq!(recipe.prefill.tokens, 30.0);
        assert_eq!(recipe.prefill.linear_flops, 6e7);
        assert_eq!(recipe.prefill.attention_flops, 76_800.0);
        assert_eq!(recipe.prefill.flops(), 60_076_800.0);
        assert_eq!(recipe.prefill.kv_bytes, 3840.0);
        assert_eq!(recipe.prefill.activation_bytes, 15_360.0);
        assert_eq!(recipe.prefill.hbm_bytes(), 1_019_200.0);
        assert_eq!(recipe.decode.tokens, 3.0);
        assert_eq!(recipe.decode_context, 16.0);
        assert_eq!(recipe.decode.linear_flops, 6e6);
        assert_eq!(recipe.decode.attention_flops, 24_576.0);
        assert_eq!(recipe.decode.flops(), 6_024_576.0);
        assert_eq!(recipe.decode.kv_bytes, 6144.0);
        assert_eq!(recipe.decode.activation_bytes, 1536.0);
        assert_eq!(recipe.decode.hbm_bytes(), 1_007_680.0);
    }

    #[test]
    fn gqa_reduces_cache_but_not_attention_operations() {
        let (mut model, workload) = fixture();
        let original = recipe(&model, &workload).unwrap();
        model.kv_heads = NonZeroU64::new(8).unwrap();
        let grouped = recipe(&model, &workload).unwrap();
        assert_eq!(grouped.kv_bytes_per_token, 512.0);
        assert_eq!(
            grouped.prefill.attention_flops,
            original.prefill.attention_flops
        );
        assert_eq!(
            grouped.decode.attention_flops,
            original.decode.attention_flops
        );
    }

    #[test]
    fn batching_amortizes_weight_traffic_per_token() {
        let (model, mut workload) = fixture();
        workload.batch = NonZeroU64::new(1).unwrap();
        let one = recipe(&model, &workload).unwrap();
        workload.batch = NonZeroU64::new(8).unwrap();
        let eight = recipe(&model, &workload).unwrap();
        assert_eq!(one.decode.weight_bytes, eight.decode.weight_bytes);
        assert!(
            eight.decode.hbm_bytes() / eight.decode.tokens
                < one.decode.hbm_bytes() / one.decode.tokens
        );
    }

    #[test]
    fn activation_traffic_uses_storage_width_for_dtype() {
        let (model, mut workload) = fixture();
        let base = 2.0 * 2.0 * 3.0 * 64.0;
        for (dtype, bytes_per_element) in [
            (Dtype::Fp4, 0.5),
            (Dtype::Fp8, 1.0),
            (Dtype::Fp16, 2.0),
            (Dtype::Bf16, 2.0),
            (Dtype::Tf32, 4.0),
            (Dtype::Fp32, 4.0),
        ] {
            workload.activation = dtype;
            let recipe = recipe(&model, &workload).unwrap();
            assert_eq!(recipe.decode.activation_bytes, base * bytes_per_element);
        }
    }

    #[test]
    fn decode_cache_traffic_scales_with_context() {
        let (model, mut workload) = fixture();
        workload.prompt_tokens = NonZeroU64::new(10).unwrap();
        let short = recipe(&model, &workload).unwrap();
        workload.prompt_tokens = NonZeroU64::new(20).unwrap();
        let long = recipe(&model, &workload).unwrap();
        assert_eq!(long.decode.kv_bytes, short.decode.kv_bytes * 26.0 / 16.0);
    }

    #[test]
    fn rejects_invalid_gqa_and_non_positive_precision_inputs() {
        let (mut model, workload) = fixture();
        model.heads = NonZeroU64::new(7).unwrap();
        assert!(recipe(&model, &workload).is_err());
        model.heads = NonZeroU64::new(8).unwrap();
        model.params = 0.0;
        assert!(recipe(&model, &workload).is_err());
        model.params = 1e6;
        model.weight_bits = f64::INFINITY;
        assert!(recipe(&model, &workload).is_err());
        model.weight_bits = 8.0;
        model.kv_bits = 0.0;
        assert!(recipe(&model, &workload).is_err());
    }
}
