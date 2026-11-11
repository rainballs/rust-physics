use rust_physics::Vec3;
use rust_physics::fluid::{Fluid, FluidSettings, PbfFluid, PbfSettings};
use std::hint::black_box;
use std::time::Instant;

fn measure(name: &str, side: i32, mut step: impl FnMut()) {
    for _ in 0..8 {
        step();
    }
    let mut samples = [0.0; 20];
    for sample in &mut samples {
        let start = Instant::now();
        step();
        *sample = start.elapsed().as_secs_f64() * 1000.0;
    }
    samples.sort_unstable_by(f64::total_cmp);
    println!(
        "{name},{},20,{:.4},{:.4}",
        side.pow(3),
        (samples[9] + samples[10]) * 0.5,
        samples[18]
    );
}

fn main() {
    println!("solver,particles,samples,median_ms,p95_ms");
    for side in [6_i32, 10, 16] {
        let mut sph = Fluid::new(FluidSettings {
            gravity: Vec3::ZERO,
            rest_density: 960.0,
            ..Default::default()
        })
        .unwrap();
        let mut pbf = PbfFluid::new(PbfSettings {
            gravity: Vec3::ZERO,
            rest_density: 960.0,
            ..Default::default()
        })
        .unwrap();
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let p = Vec3::new(f64::from(x), f64::from(y), f64::from(z)) * 0.16;
                    sph.push(p, Vec3::ZERO);
                    pbf.push(p, Vec3::ZERO);
                }
            }
        }
        measure("SPH", side, || {
            black_box(sph.step(1.0 / 240.0, None));
        });
        measure("PBF (5 projections)", side, || {
            black_box(pbf.step(1.0 / 240.0, None));
        });
        assert!(
            sph.positions()
                .iter()
                .chain(pbf.positions())
                .all(|p| p.is_finite())
        );
    }
}
