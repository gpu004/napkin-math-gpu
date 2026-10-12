use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("napkin build.rs: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?;
    let profiles_dir = Path::new(&manifest_dir).join("../../profiles");
    println!("cargo:rerun-if-changed={}", profiles_dir.display());
    let mut entries = Vec::new();
    let read_dir = fs::read_dir(&profiles_dir).map_err(|e| e.to_string())?;
    for entry in read_dir {
        let path = entry.map_err(|e| e.to_string())?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(id) = name.strip_suffix(".measured.json") {
            entries.push((id.to_owned(), path.clone()));
        }
    }
    entries.sort();
    let mut generated = String::from("const MEASURED_SRC: &[(&str, &str)] = &[\n");
    for (id, path) in &entries {
        writeln!(
            generated,
            "    ({id:?}, include_str!({:?})),",
            path.display().to_string()
        )
        .map_err(|e| e.to_string())?;
    }
    generated.push_str("];\n");
    let out_dir = env::var("OUT_DIR").map_err(|e| e.to_string())?;
    fs::write(Path::new(&out_dir).join("measured.rs"), generated).map_err(|e| e.to_string())?;
    Ok(())
}
