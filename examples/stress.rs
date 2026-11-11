mod stress_scene;
use std::time::Instant;
use stress_scene::soak;

fn main() {
    let mut args = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>().expect("integer argument"));
    let runs = args.next().unwrap_or(8);
    let frames = args.next().unwrap_or(600);
    let start = Instant::now();
    let mut contacts = 0;
    let mut impacts = 0;
    let first_seed = args.next().unwrap_or(0);
    let verbose = std::env::var_os("RUST_PHYSICS_STRESS_VERBOSE").is_some();
    let trace = std::env::var_os("RUST_PHYSICS_STRESS_TRACE").is_some();
    for seed in first_seed..first_seed + runs {
        let (c, i) = soak(seed as u64, frames, trace);
        contacts += c;
        impacts += i;
        let completed = seed + 1 - first_seed;
        if verbose || completed.is_multiple_of(8) || completed == runs {
            println!("{} / {runs} seeds, {} steps", completed, completed * frames);
        }
    }
    println!(
        "{contacts} contact points, {impacts} swept impacts, {:.2} seconds",
        start.elapsed().as_secs_f64()
    );
}
