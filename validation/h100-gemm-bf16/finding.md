Within tolerance. The run held 1275–1350 MHz SM clocks at 695–701 W, the
software power cap, which is the same sustained state the recalibrated profile
measured (#27: 691 TFLOP/s bf16 under `sw_power_cap`). cuBLAS reached
7.70 TFLOP / 11.22 ms = 686 TFLOP/s on this shape, so the 8192³ calibration
rate carried over to a 16384 × 28672 × 8192 GEMM with no visible wave or tile
quantization loss. Before #27 the profile said 734 TFLOP/s, which would have
given a ratio of about 1.07, still within the band but biased fast.

The observation host ran driver 610.57.04 while the profile was measured on
580.95.05; both are the same H100 80GB HBM3 model, so this case also shows the
GEMM rate did not move across that driver change.
