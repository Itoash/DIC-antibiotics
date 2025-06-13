use core::num;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ac_processing::utils::sos::{bandpass_swapped};
use ndarray::{Array3};


fn generate_time_series_grid(num_points: usize,y:usize,x:usize) -> Array3<f32> {
        use std::f32::consts::PI;
        // Generate the time series signal
        let signal: Vec<f32> = (0..num_points)
            .map(|i| (i as f32 * 2.0 * PI / (num_points - 1) as f32).sin())
            .collect();
        
        // Create Array3 with shape [time, height, width] = [num_points, 2, 2]
        let mut array = Array3::<f32>::zeros((y, x, num_points));
        
        // Fill the array - repeat the signal across the 2x2 spatial grid
        for t in 0..num_points {
            for i in 0..y{
                for j in 0..x {
                    array[[i,j,t]] = signal[t];
                }
            }
        }
        
        array
    }
fn bench_bandpasses(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_array3");
    
    group.measurement_time(std::time::Duration::from_secs(20)); // Increase measurement time
    group.sample_size(10); // Decrease sample count
    group.warm_up_time(std::time::Duration::from_secs(5)); // Warm-up time
    // Test different sizes
    let sizes = vec![
            (400,256,256),
            (400,512,512),
            (400,1024,1024),
             (400,2048,2048),
             (400,4096,4096)
         

    ];
    
    for (time_points,y,x) in sizes {
        
        group.bench_function(
            BenchmarkId::new("bandpass_swapped", format!("{}t_{}y_{}x", time_points, y, x)),
            |b| {
                b.iter_with_setup(
                    || generate_time_series_grid(time_points, y, x), // Setup: create fresh data
                    |mut data| black_box(bandpass_swapped(&mut data, 0.1, 1f32, 2f32)) // Benchmark: use data
                )
            },
        );
    

        
        
    }
    
    group.finish();
}

criterion_group!(benches,bench_bandpasses);
criterion_main!(benches);
// use criterion::{black_box, criterion_group,criterion_main, Criterion, BenchmarkId};
// use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut1};

// // Test data generator
// fn generate_test_data(n_samples: usize, n_sections: usize) -> (Array1<f32>, Array2<f32>) {
//     let signal = Array1::from_vec((0..n_samples).map(|i| (i as f32).sin()).collect());
    
//     // Generate realistic SOS coefficients
//     let mut sos = Array2::zeros((n_sections, 6));
//     for s in 0..n_sections {
//         sos[[s, 0]] = 1.0;  // b0
//         sos[[s, 1]] = 0.5;  // b1  
//         sos[[s, 2]] = 0.25; // b2
//         sos[[s, 3]] = 1.0;  // a0 (normalized)
//         sos[[s, 4]] = -0.3; // a1
//         sos[[s, 5]] = 0.1;  // a2
//     }
    
//     (signal, sos)
// }

// // Original version with loop
// #[inline(always)]
// fn filter_with_loop(signal: &mut ArrayViewMut1<f32>, sos: ArrayView2<f32>) {
//     let n_sections = sos.nrows();
//     let mut z1 = vec![0.0f32; n_sections];
//     let mut z2 = vec![0.0f32; n_sections];
    
//     for x in signal.iter_mut() {
//         let mut sample = *x;
//         for s in 0..n_sections {
//             let b0 = sos[[s, 0]];
//             let b1 = sos[[s, 1]];
//             let b2 = sos[[s, 2]];
//             let a1 = sos[[s, 4]];
//             let a2 = sos[[s, 5]];
            
//             let y = b0.mul_add(sample, z1[s]);
//             z1[s] = b1.mul_add(sample, a1.mul_add(-y, z2[s]));
//             z2[s] = b2.mul_add(sample, -a2 * y);
//             sample = y;
//         }
//         *x = sample;
//     }
// }

// // Unrolled version for 2 sections
// #[inline(always)]
// fn filter_unrolled_2_sections(signal: &mut ArrayViewMut1<f32>, sos: ArrayView2<f32>) {
//     let b0_0 = sos[[0, 0]]; let b1_0 = sos[[0, 1]]; let b2_0 = sos[[0, 2]];
//     let a1_0 = sos[[0, 4]]; let a2_0 = sos[[0, 5]];
//     let b0_1 = sos[[1, 0]]; let b1_1 = sos[[1, 1]]; let b2_1 = sos[[1, 2]];
//     let a1_1 = sos[[1, 4]]; let a2_1 = sos[[1, 5]];
    
//     let mut z1_0 = 0.0f32; let mut z2_0 = 0.0f32;
//     let mut z1_1 = 0.0f32; let mut z2_1 = 0.0f32;
    
//     for x in signal.iter_mut() {
//         let input = *x;
        
//         // Section 0
//         let y0 = b0_0.mul_add(input, z1_0);
//         z1_0 = b1_0.mul_add(input, a1_0.mul_add(-y0, z2_0));
//         z2_0 = b2_0.mul_add(input, -a2_0 * y0);
        
//         // Section 1  
//         let y1 = b0_1.mul_add(y0, z1_1);
//         z1_1 = b1_1.mul_add(y0, a1_1.mul_add(-y1, z2_1));
//         z2_1 = b2_1.mul_add(y0, -a2_1 * y1);
        
//         *x = y1;
//     }
// }

// // Pre-extracted coefficients version (likely more impactful)
// #[inline(always)]
// fn filter_preextracted(signal: &mut ArrayViewMut1<f32>, sos: ArrayView2<f32>) {
//     let n_sections = sos.nrows();
    
//     // Pre-extract coefficients
//     let coeffs: Vec<(f32, f32, f32, f32, f32)> = (0..n_sections)
//         .map(|s| (sos[[s, 0]], sos[[s, 1]], sos[[s, 2]], sos[[s, 4]], sos[[s, 5]]))
//         .collect();
    
//     let mut z1 = vec![0.0f32; n_sections];
//     let mut z2 = vec![0.0f32; n_sections];
    
//     for x in signal.iter_mut() {
//         let mut sample = *x;
//         for (s, &(b0, b1, b2, a1, a2)) in coeffs.iter().enumerate() {
//             let y = b0.mul_add(sample, z1[s]);
//             z1[s] = b1.mul_add(sample, a1.mul_add(-y, z2[s]));
//             z2[s] = b2.mul_add(sample, -a2 * y);
//             sample = y;
//         }
//         *x = sample;
//     }
// }

// fn bench_loop_unrolling(c: &mut Criterion) {
//     let mut group = c.benchmark_group("loop_unrolling_comparison");
    
//     // Test different scenarios
//     let scenarios = vec![
//         (1000, 1, "1k_samples_1_section"),
//         (1000, 2, "1k_samples_2_sections"),  
//         (1000, 4, "1k_samples_4_sections"),
//         (10000, 2, "10k_samples_2_sections"),
//         (100000, 2, "100k_samples_2_sections"),
//     ];
    
//     for (n_samples, n_sections, name) in scenarios {
//         let (test_signal, test_sos) = generate_test_data(n_samples, n_sections);
        
//         // Test original loop version
//         group.bench_function(
//             BenchmarkId::new("with_loop", name),
//             |b| b.iter_with_setup(
//                 || test_signal.clone(),
//                 |mut signal| {
//                     let mut view = signal.view_mut();
//                     black_box(filter_with_loop(&mut view, test_sos.view()))
//                 }
//             )
//         );
        
//         // Test pre-extracted coefficients (likely more impactful optimization)
//         group.bench_function(
//             BenchmarkId::new("preextracted", name), 
//             |b| b.iter_with_setup(
//                 || test_signal.clone(),
//                 |mut signal| {
//                     let mut view = signal.view_mut();
//                     black_box(filter_preextracted(&mut view, test_sos.view()))
//                 }
//             )
//         );
        
//         // Test unrolled version (only for 2 sections)
//         if n_sections == 2 {
//             group.bench_function(
//                 BenchmarkId::new("unrolled_2", name),
//                 |b| b.iter_with_setup(
//                     || test_signal.clone(),
//                     |mut signal| {
//                         let mut view = signal.view_mut();
//                         black_box(filter_unrolled_2_sections(&mut view, test_sos.view()))
//                     }
//                 )
//             );
//         }
//     }
    
//     group.finish();
// }
// criterion_group!(benches, bench_loop_unrolling);
// criterion_main!(benches);