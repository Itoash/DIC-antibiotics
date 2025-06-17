


pub mod utils{
    pub mod loading{
        // pub fn sort_files(path:&str)->(){ }
        // pub fn load_images(path:&str)->(){}

    }

    pub mod processing {
        use ndarray::{Array3,ArrayView3,Array1,Axis,Zip};
        use rayon::prelude::*;
        
        
        #[inline(always)]
        pub fn cubic_hermite_interp(a:f32,b:f32,c:f32,d:f32,t:f32)->f32{
            let a_out = -a/2.0f32 + (3.0f32*b)/2.0f32 - (3.0f32*c)/2.0f32 + d/2.0f32;
            let b_out = a - (5.0f32*b)/2.0f32 + 2.0f32*c - d / 2.0f32;
            let c_out = -a/2.0f32 + c/2.0f32;
            let d_out = b;
            a_out*t*t*t + b_out*t*t +c_out*t +d_out
        }
        #[inline(always)]
        pub fn clamp_index(index: usize,size:usize) -> usize {
            index.max(0).min(size-1)
        }

        #[inline]
        pub fn interpolate_time_legacy(mut stack:Array3<f32>,old_dt:f32, new_dt: f32)->Array3<f32> {
            
            let (height,width,ntimes) = stack.dim();
            println!("Made it to line 2 calculation!");
            let times: Array1<f32> = (0..ntimes)
                            .map(|i| i as f32 * old_dt)
                            .collect::<Array1<f32>>();
            let total_time = times[times.len()-1];
            
            let new_times = Array1::range(0.0, total_time + new_dt/2.0, new_dt);
            let new_time_points = new_times.len(); // +1 to include endpoint
            let scale_factor = (times.len() - 1) as f32 / (new_time_points - 1) as f32;
            
            let interp_params: Vec<(usize, usize,usize,usize, f32)> = (0..new_time_points)
                .into_par_iter()
                .map(|i| {
                    let x = i as f32*scale_factor;

                    let index = x as usize;
                    let t = x - x.floor();
                    let idx_a = clamp_index(index-1,ntimes);
                    let idx_b = clamp_index(index,ntimes);
                    let idx_c = clamp_index(index+1,ntimes);
                    let idx_d = clamp_index(index+2,ntimes);
                    (idx_a, idx_b, idx_c,idx_d,t)
                })
                .collect();
                
                let result = stack
                .lanes_mut(Axis(2))
                .into_iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|signal| {
                    let mut new_signal = Array1::zeros(new_time_points);
                    for (i, &(idx_a, idx_b, idx_c,idx_d,t)) in interp_params.iter().enumerate() {
                        new_signal[i] = cubic_hermite_interp(signal[idx_a],signal[idx_b],signal[idx_c],signal[idx_d],t);
                    }
                    new_signal
                })
                .collect::<Vec<_>>();
            
            to_array3(result,ntimes,height,width)
            
        }
        #[inline(always)]
        fn to_array3(stack:Vec<Array1<f32>>,ntimes:usize,height:usize,width:usize)->Array3<f32>{
            let mut result = Array3::zeros((ntimes, height, width));
                    
            // Parallel iteration over spatial coordinates
            Zip::indexed(&mut result.view_mut())
                .par_for_each(|(t, y, x), value| {
                    let pixel_index = y * width + x;
                    if pixel_index < stack.len() {
                        *value = stack[pixel_index][t];
                    }
                });
                    
            result
        }
       
        // Main version
        #[inline]
        pub fn interpolate_time(stack: &ArrayView3<f32>, old_dt: f32, new_dt: f32) -> Array3<f32> {
            //((y,x,t))
            let ( height, width,ntimes) = stack.dim();
            let total_time = (ntimes - 1) as f32 * old_dt;
            //new N_times
            let new_time_points = ((total_time / new_dt).floor() as usize) + 1;
            //Stretch factor
            let scale_factor = (ntimes - 1) as f32 / (new_time_points - 1) as f32;
            
            // Pre-compute interpolation parameters  
            let interp_params: Vec<(usize, usize, usize, usize, f32)> = (0..new_time_points)
                .map(|i| {
                    let x = i as f32 * scale_factor;
                    let index = x as usize;
                    let t = x - x.floor();
                    let idx_a = clamp_index(index.saturating_sub(1), ntimes);
                    let idx_b = clamp_index(index, ntimes);
                    let idx_c = clamp_index(index + 1, ntimes);
                    let idx_d = clamp_index(index + 2, ntimes);
                    (idx_a, idx_b, idx_c, idx_d, t)
                })
                .collect();
            
            // Create flattened result vector
            let total_elements = new_time_points * height * width;
            let result_vec: Vec<f32> = (0..total_elements)
                .into_par_iter()
                .map(|flat_idx| {
                    let y = flat_idx / (width * new_time_points);
                    let remainder = flat_idx % (width * new_time_points);
                    let x = remainder / new_time_points;
                    let t = remainder % new_time_points;
                    
                    let (idx_a, idx_b, idx_c, idx_d, interp_t) = interp_params[t];
                    
                    cubic_hermite_interp(
                        stack[[ y, x,idx_a]],
                        stack[[ y, x,idx_b]],
                        stack[[ y, x,idx_c]],
                        stack[[ y, x,idx_d]],
                        interp_t
                    )
                })
                .collect();
            
            // Reshape back to 3D
            Array3::from_shape_vec(( height, width,new_time_points), result_vec)
                .expect("Shape mismatch in interpolation result")
        }
    }

    pub mod filters{
        use num_traits::{Num,Zero,Float};
        use std::ops::{AddAssign,Neg};
        use num_complex::Complex;
        use std::f32::consts::PI;
        use ndarray::Array2;
        // trait for easy check if is real
        trait IsReal {
            fn is_real(&self) -> Vec<bool>;
        }

        // Scalar impl
        impl<T: Float> IsReal for Complex<T> {
            #[inline(always)]
            fn is_real(&self) -> Vec<bool> {
                vec![self.im == T::zero()]
            }
        }
        // Slice impl
        impl<T: Float> IsReal for [Complex<T>] {
            #[inline(always)]
            fn is_real(&self) -> Vec<bool> {
                self.iter().map(|c| c.im == T::zero()).collect()
            }
        }
        /// Generate trivial poles/zeros for prototype BUtterworth filter
        fn buttap(N:isize)->(Vec<Complex<f32>>,Vec<Complex<f32>>,f32){
            assert!(N>0);
            let z = Vec::new();
            let m :Vec<isize>= (-N+1..N).step_by(2).collect();
            let p = (0..m.len())
                    .map(|i| -Complex::new(0f32,m[i] as f32*PI/(2f32*N as f32)).exp())
                    .collect();
            let k = 1f32;
            (z,p,k)
        }
        /// Scale low-pass zeros/poles and gain to unit circle for low-pass
        fn lp2lp_zpk(z:&[Complex<f32>],p:&[Complex<f32>],k:f32,wo:f32)->(Vec<Complex<f32>>,Vec<Complex<f32>>,f32){
            // get relative degree
            let degree = _relative_degree(z, p);

            //scale to unit circl;
            let z_lp:Vec<Complex<f32>> = z.iter().map(|x| x*wo).collect();
            let p_lp:Vec<Complex<f32>> = p.iter().map(|x| x*wo).collect();
            let k_lp = k* wo.powi(degree);

            (z_lp,p_lp,k_lp)
        } 
        /// Convert low-pass/prototype zeros/poles arrays, as well as gain, to band-pass equivalents
        fn lp2bp_zpk(z:&[Complex<f32>],p:&[Complex<f32>],k:f32,wo:f32,bw:f32)->(Vec<Complex<f32>>,Vec<Complex<f32>>,f32){
            //get relative degree
            let degree = _relative_degree(z, p);
            //scale to bandwidth
            let z_lp:Vec<Complex<f32>> = z.iter().map(|x| x*bw/2f32).collect();
            let p_lp:Vec<Complex<f32>> = p.iter().map(|x| x*bw/2f32).collect();
    
            let wo_c = Complex::new(wo,0f32);
            
            // Scale poles/zeros from baseband to -wo,+wo and duplicate (complex conjugate, remember?)
            let mut z_bp:Vec<Complex<f32>>  = z_lp.clone().
                        into_iter()
                        .map(|z|z+(z*z-wo_c*wo_c).sqrt())
                        .collect();
            let z_bp_right:Vec<Complex<f32>>  = z_lp.clone().
                        into_iter()
                        .map(|z|z-(z*z-wo_c*wo_c).sqrt())
                        .collect();
            z_bp.extend(z_bp_right);
            
            let mut p_bp:Vec<Complex<f32>>  = p_lp.clone().
                        into_iter()
                        .map(|z|z+(z*z-wo_c*wo_c).sqrt())
                        .collect();
            
            
            let p_bp_right:Vec<Complex<f32>>  = p_lp.clone().
                        into_iter()
                        .map(|z|z-(z*z-wo_c*wo_c).sqrt())
                        .collect();
            p_bp.extend(p_bp_right);
            
            let z = Complex::new(6.5823f32,-0.124f32);
            
            //Add zeros at infinity
            z_bp.extend(vec![Complex::new(0f32,0f32);degree as usize]);

            let k_bp = k*(bw.powi(degree));
            
            (z_bp,p_bp,k_bp)
        }     
        fn _relative_degree<T>(z:&[T],p:&[T])->i32{
            let degree:isize = p.len() as isize-z.len() as isize;
            assert!(degree>0);
            degree as i32
        }
        /// Bilinear transform of zeros/poles/gain, from continuous s-plane to discrete z-plane
        fn bilinear_zpk(z:&[Complex<f32>],p:&[Complex<f32>],k:f32,fs:f32)->(Vec<Complex<f32>>,Vec<Complex<f32>>,f32){
            // Check sampling freq
            assert!(fs>0f32);
            // Get difference in poles/zeroes
            let degree = _relative_degree(z, p);
            // Easy precompute
            let fs2 = 2f32*fs;
            //Bilinear transform of zeros and poles
            let mut z_z:Vec<Complex<f32>> = z
                                            .iter()
                                            .map(|s| (fs2+s)/(fs2-s))
                                            .collect();
            let p_z:Vec<Complex<f32>> = p
            .iter()
            .map(|s| (fs2+s)/(fs2-s))
            .collect();
            // Zeroes at inf get moved to cutoff
            z_z.extend(vec![Complex::new(-1f32,0f32);degree as usize]);

            // Compute gain
            let num: Complex<f32> = z.iter().map(|zi| fs2 - zi).product();
            let den: Complex<f32> = p.iter().map(|pi| fs2 - pi).product();
            let k_z = k*(num / den).re;

            (z_z,p_z,k_z)
        }
        // Check if array only has complex conjugates, then return complex and real parts without complex conjugate duplicates
        fn _cplxreal( z:&[Complex<f32>])->(Vec<Complex<f32>>,Vec<Complex<f32>>){
            if z.is_empty(){(z.to_vec(), z.to_vec())}
            else{
                let has_nan = z.iter().any(|x| x.re.is_nan() || x.im.is_nan());

                if has_nan {
                    println!("NaN values detected in vector:");
                    for (i, item) in z.iter().enumerate() {
                        println!("  [{}]: re={}, im={}", i, item.re, item.im);
                    }
                }
                //Tolerance
                let tol = 100.0 * f32::EPSILON;
                //sort by real first then imaginary
                let mut z = z.to_vec();
                z.sort_by(|a, b| {
                    let key_a = (a.re, a.im.abs());
                    let key_b = (b.re, b.im.abs());
                    key_a.partial_cmp(&key_b).unwrap_or_else(|| panic!("Failed to compare values: key_a={:?}, key_b={:?}, a={:?}, b={:?}", 
        key_a, key_b, a, b))
                });

                // Compare each element's imag part to tol * its own abs to get "real" values
                let zr = z.iter().filter(|x| x.im.abs() <= tol * x.norm()).cloned().collect::<Vec<_>>();
                
                if zr.len() == z.len(){
                    return(Vec::new(),zr)
                }
                //Positive/negative imaginary components
                let mut zp = z.iter().filter(|x| x.im.abs() > tol * x.norm() && x.im >0f32).cloned().collect::<Vec<_>>();
                let mut zn = z.iter().filter(|x| x.im.abs() > tol * x.norm() && x.im <0f32).cloned().collect::<Vec<_>>();
                assert!(zp.len()==zn.len());
                //Bool vec of wether differences are less than tolerance
                let same_real: Vec<bool> = zp.windows(2)
                .map(|w| (w[1].re - w[0].re).abs() <= tol * w[0].norm())
                .collect();
                // Padded with false at ends
                let mut padded = Vec::with_capacity(same_real.len() + 2);
                padded.push(false);
                padded.extend(same_real.iter().cloned());
                padded.push(false);

                //Assign score: (false,true)->-1 run starts; (true,false)->+1 run ends; _-> no change (0)
                let diffs: Vec<i8> = padded.windows(2)
                    .map(|w| match (w[0], w[1]) {
                        (false, true) => 1,
                        (true, false) => -1,
                        _ => 0,
                    })
                    .collect();
                // Collect start indices
                let run_starts: Vec<usize> = diffs.iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v > 0 { Some(i) } else { None })
                    .collect::<Vec<usize>>();
                // Collect stop indices
                let run_stops = diffs.iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v < 0 { Some(i) } else { None })
                    .collect::<Vec<usize>>();
                // Sort sub-slice in-place by |imag| using runs
                for (&start, &stop) in run_starts.iter().zip(run_stops.iter()) {
                    
                    zp[start..=stop].sort_by(|a, b| a.im.abs().partial_cmp(&b.im.abs()).unwrap());
                    zn[start..=stop].sort_by(|a, b| a.im.abs().partial_cmp(&b.im.abs()).unwrap());
                }
                let unmatches:Vec<(Complex<f32>,Complex<f32>)> = zp.iter().zip(zn.iter())
                    .filter(|tup| (*tup.0 - tup.1.conj()).norm() > tol * tup.1.norm())
                    .map(|(a, b)| ((*a, *b)))
                    .collect();
                if !unmatches.is_empty(){
                    panic!("Non-conjugate elements found in _cplxreal!");
                }
                let zc = zp.iter().zip(zn.iter()).map(|(p,n)| (p+n.conj()).unscale(2f32)).collect();
                
                (zc,zr)
            }
        }
        /// Get index of highest abs value for s-plane point
        fn idx_worst(p: &[Complex<f32>]) -> usize {
            let values: Vec<f32> = p.iter().map(|x| (1f32 - x.norm()).abs()).collect();
            let mut min_index = 0;
            let mut min_value = f32::INFINITY;
            for (i, &val) in values.iter().enumerate() {
                if val < min_value {
                    min_value = val;
                    min_index = i;
                }
            }
            min_index
        }
        /// enum for _nearest function, specify options
        enum Nearest{
            Real,
            Complex,
            Any,
        }
        /// math helper function for convolution
        fn convolve_full<T>(a: &[T], b: &[T]) -> Vec<T>
        where
            T: Num + Copy + Zero + AddAssign,
        {
            let n = a.len();
            let m = b.len();
            let mut result = vec![T::zero(); n + m - 1];

            for (i, &ai) in a.iter().enumerate() {
                for (j, &bj) in b.iter().enumerate() {
                    result[i + j] += ai * bj;
                }
            }
            result
        }
        /// Get nearest value in complex array, wrt given to parameter
        /// specify some restrictions regarding result (has to be real/complex, or any)
        fn _nearest_real_complex_idx(from:&[Complex<f32>],to:Complex<f32>,which:Nearest)->usize{
            let absolute_value:Vec<f32> = from.iter().map(|f|(f-to).norm()).collect();
            let mut order = (0..absolute_value.len())
                .collect::<Vec<usize>>();

            order.sort_by(|&i, &j| absolute_value[i].partial_cmp(&absolute_value[j]).unwrap());
            match which{
                Nearest::Any=>{order[0]},
                Nearest::Real=>{
                    order.iter()
                .copied()
                .find(|&i| from[i].im == 0.0)
                .unwrap()}
                Nearest::Complex=>{order.iter()
                .copied()
                .find(|&i| from[i].im != 0.0)
                .unwrap()}

            }

        }
        /// Math helper function that approximatesthe coeffs of a polynomial, given its zeros. Very neat convolution trick
        fn _poly<T>(zeros:&[T])->Vec<T>
        where
            T: Num + Copy + Zero + AddAssign + Neg<Output = T>,
        {
            if zeros.is_empty() {
                vec![T::one()]
            } else {
                let mut res = vec![T::one()];
                for z in zeros.iter() {
                    let kernel = vec![T::one(), -*z];
                    res = convolve_full(&res, &kernel);
                }
                res
            }
        }
        // format second order sections and collect
        fn _single_zpksos(z:&[Complex<f32>],p: &[Complex<f32>],k:f32)->Vec<Complex<f32>>{
            let b:Vec<_> =  _poly(z).iter().map(|z|z*k).collect();
            let a:Vec<_> = _poly(p);
            let mut sos = vec![Complex::new(0f32,0f32);6];
            (3-b.len()..3)
            .zip(0..b.len())
            .for_each(|(i,j)| sos[i]=b[j]);
            (6-a.len()..6)
            .zip(0..a.len())
            .for_each(|(i,j)| sos[i]=a[j]);
            sos
        }
        // main logic for extracting sos from zeros poles and gain; hellish
        fn zpk2sos(z:&[Complex<f32>],p: &[Complex<f32>],k:f32)->Vec<Vec<Complex<f32>>>{
            // early return
            if z.len()==p.len() && p.is_empty(){
                return vec![vec![
                    Complex::new(k, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                ]];
            }
            
            //ensure same number of zeros/poles -> copy over
            let mut p_new = (*p).to_vec();
            
            p_new.extend(vec![Complex::new(0f32,0f32); (z.len() - p.len()).max(0)]);
            
            let mut z_new = (*z).to_vec();
            z_new.extend(vec![Complex::new(0f32,0f32); (z.len() - p.len()).max(0)]);

            let n_sections = p_new.len().max(z_new.len()+1)/2;
            if p_new.len() % 2 == 1{
                p_new.push(Complex::new(0f32,0f32));
                z_new.push(Complex::new(0f32,0f32));
            }
            assert_eq!(p_new.len(),z_new.len());

            //Get complex conjugate pairs
            
            let (zc, zr) = _cplxreal(&z_new);
            let mut z_new = zc.into_iter().chain(zr).collect::<Vec<_>>();
        
            let (pc,pr) = _cplxreal(&p_new);
            let mut p_new = pc.into_iter().chain(pr).collect::<Vec<_>>();

            let mut sos = vec![vec![Complex::new(0f32,0f32);6];n_sections];

            for si in (0..n_sections).rev() {
                //si is 1 larger than python here
                //Get worst index
                let p1_idx = idx_worst(&p_new);
                let p1 = p_new[p1_idx];
                p_new.remove(p1_idx);
                //Special case: last remaining pole
                let sum_reals = p_new.iter().filter(|x| x.im == 0.).cloned().collect::<Vec<Complex<f32>>>().len();
                if p1.im == 0. && sum_reals ==0{
                    if !z_new.is_empty(){
                        let z1_idx = _nearest_real_complex_idx(&z_new, p1, Nearest::Real);
                        let z1 = z_new[z1_idx];
                        z_new.remove(z1_idx);
                        sos[si] = _single_zpksos(&z_new, &p_new, k);
                    }
                    else{
                        sos[si] = _single_zpksos(&Vec::new(), &[p1], k);
                    }
                // special case one real pole and zero, not equal numer of poles/zeros, must pair with complex 0
                } else if p_new.len()+1 == z_new.len() &&
                            p1.im != 0f32 && 
                            p_new.is_real().iter().filter(|x| **x).collect::<Vec::<_>>().len() == 1 &&
                            z_new.is_real().iter().filter(|x| **x).collect::<Vec::<_>>().len() == 1 {
                        
                        let z1_idx = _nearest_real_complex_idx(&z_new, p1, Nearest::Complex);
                        let z1 = z_new[z1_idx];
                        z_new.remove(z1_idx);
                        sos[si] = _single_zpksos(&[z1, z1.conj()], &[p1, p1.conj()], 1f32);
                } else{
                    let mut p2 = Complex::new(0f32,0f32);
                    if p1.is_real()[0]{ // real pole, get another real pole (worst one)
                        let p_reals_bool = p_new.is_real();
                        let prealidx:Vec<usize> = (0..p_reals_bool.len()).filter(|&i|p_reals_bool[i]).collect();
                        let p_reals: Vec<Complex<f32>> = prealidx.iter().map(|&i| p_new[i]).collect();
                        let p2_idx = prealidx[idx_worst(&p_reals)];
                        p2 = p_new[p2_idx];
                        p_new.remove(p2_idx);
                    } else { // just get conjugate, we've guaranteed it exists
                        p2 = p1.conj();
                    }
                    // find closest zeros to selected poles
                    if !z_new.is_empty(){
                        let z1_idx = _nearest_real_complex_idx(&z_new, p1, Nearest::Any);
                        let z1 = z_new[z1_idx];
                        z_new.remove(z1_idx);
                        if  !(z1.is_real()[0]){ // not real zero - use conj
                            sos[si] = _single_zpksos(&[z1, z1.conj()], &[p1, p2], 1f32);
                        } else if !z_new.is_empty(){
                            let z2_idx = _nearest_real_complex_idx(&z_new, p1,Nearest::Real);
                            let z2 = z_new[z2_idx];
                            assert!(z2.is_real()[0]);
                            z_new.remove(z2_idx);
                            sos[si] = _single_zpksos(&[z1, z2], &[p1, p2], 1f32);
                        }
                        else{
                            sos[si] = _single_zpksos(&[z1], &[p1, p2], 1f32);
                        }
                    }else{ // no more zeros
                        sos[si] = _single_zpksos(&[], &[p1, p2], 1f32);
                    }
                    }
                }
                assert_eq!(z_new.len(),p_new.len());
                assert_eq!(z_new.len(),0); // consumed all poles and zeros

                //put gain in first sos
                sos[0].iter_mut().take(3).for_each(|x| *x *= k);
                sos

            }

        

        // Exposed convenience function for constructing a bandpass butterworth filter
        #[inline]
        pub fn butter_bandpass(order:usize,cutoffs:&[f32],fs:f32)->Array2<f32>{
            // do some checks
            assert!(cutoffs[0]<cutoffs[1]);
            assert!(!cutoffs.iter().filter(|&x| *x < fs/2f32).collect::<Vec<_>>().is_empty());
            // normalize cutoffs
            let Wn:Vec<f32> = cutoffs.iter().map(|&x| x/(fs/2f32)).collect();
            
            // get zeros/poles/gain for this order of filter from Butter formula
            let (z,p,k) = buttap(order.try_into().unwrap());
            // Warp frequencies for digital filter design
            let fs = 2f32;
            let warped: Vec<f32> = Wn.iter().map(|&x| 2f32 * fs * (PI * x / fs).tan()).collect();
            
            let bw = warped[1]-warped[0];//bandwidth
            let wo = (warped[0]*warped[1]).sqrt(); //omega-0, center frequency

            let (z,p,k) = lp2bp_zpk(&z,&p,k,wo,bw); // transform from lowpass prototype to bandpass
            
            let (z,p,k) = bilinear_zpk(&z, &p, k, fs); //bilinear to get to z-domain (discrete) from s-domain(continuous)
            
            let sos = zpk2sos(&z,&p,k);
            let isreal = sos.iter().map(|x| x.is_real()).collect::<Vec<_>>();
            let complex_no = isreal.iter().flatten().filter(|&x| !x).collect::<Vec<_>>().len();
            assert_eq!(complex_no,0);
            let final_sos = 
            sos.into_iter()
            .map(|inner|inner.into_iter().map(|c|c.re).collect::<Vec<_>>())
            .collect::<Vec<Vec<_>>>();
            //println!("{:?}",final_sos);
            let nrows = final_sos.len();
            let ncols = final_sos[0].len();
            let flat: Vec<f32> = final_sos.into_iter().flatten().collect();
            let mut sos = Array2::from_shape_vec((nrows, ncols), flat).unwrap();
            for n in 0..sos.nrows(){
                let a_0 = sos[[n,3]];
                if a_0 != 0f32 && a_0 != 1f32{
                    for i in 0..6 {
                        sos[[n,i]] /= a_0;
                    }
                }
            }

            sos
            

        }
    }

    pub mod sos{
        use ndarray::{ArrayViewMut1,ArrayView2,Array2};
        use ndarray::parallel::prelude::*;
        
        /// Second-Order Section coefficients [b0, b1, b2, a0, a1, a2]
        /// Note: a0 is typically 1.0 and sometimes omitted in representations
        #[derive(Debug, Clone)]
        #[repr(C)]

        pub struct SosFilter {
            /// Second-order sections coefficients [n_sections, 6]
            /// Each row: [b0, b1, b2, a0, a1, a2] where a0 is typically 1.0
            sos: Array2<f32>,
            /// Internal filter state [n_sections, 2] 
            /// Each row: [z1, z2] for the delay elements
            zi: Array2<f32>,
        }

        impl SosFilter {
            /// Create a new SOS filter from coefficients
            /// 
            /// # Arguments
            /// * `sos` - Second-order sections as Array2 with shape [n_sections, 6]
            /// 
            /// # Returns
            /// New SosFilter with zero initial conditions
            pub fn new(sos: Array2<f32>) -> Self {
                let n_sections = sos.nrows();
                let zi = Array2::zeros((n_sections, 2));
                
                
                Self { sos, zi }
            }
            
            
            /// Create SOS filter with custom initial conditions
            pub fn with_initial_conditions(sos: Array2<f32>, zi: Array2<f32>) -> Self {
                assert_eq!(sos.nrows(), zi.nrows(), "Number of sections must match");
                assert_eq!(zi.ncols(), 2, "State must have 2 columns [z1, z2]");
                
                Self { sos, zi }
            }
            
            /// Reset filter state to zero
            pub fn reset(&mut self) {
                self.zi.fill(0.0);
            }
            
            /// Get current filter state (useful for debugging or state inspection)
            pub fn state(&self) -> ArrayView2<f32> {
                self.zi.view()
            }
            
            /// Apply SOS filter to signal in-place
            /// 
            /// This is the most efficient version, modifying the input signal directly
            #[inline(always)]
            
            pub fn filter_inplace(&mut self, signal: &mut ArrayViewMut1<f32>) {
                let sos = self.sos.view();
                let mut zi = self.zi.view_mut();
                let n_samples = signal.len();
                let n_sections = self.sos.nrows();
                // (0..n_samples).into_iter().for_each(|n|{
                //     let mut x = signal[n];
                //     (0..n_sections).into_iter().for_each(|s|{
                //         let y = x.mul_add(self.sos[[s,0]],self.zi[[s,0]]);
                //         // Update state (order matters!)
                //         // self.zi[[s, 0]] = x * b1 + z2 - a1 * y;
                //         // self.zi[[s, 1]] = x * b2 - a2 * y;
                //         self.zi[[s, 0]] = x.mul_add(self.sos[[s,1]],y.mul_add(-self.sos[[s,4]],self.zi[[s,1]]));
                //         self.zi[[s,1]] = x.mul_add(self.sos[[s,2]],-self.sos[[s,5]]*y);
                //         // Output of this section becomes input to next section
                //         x = y;
                //     });
                //     signal[n] = x;
                // });
                for n in 0..n_samples {
                    let mut x = signal[n]; // Start with original input
                    
                    for s in 0..n_sections {
                        // // Load coefficients
                        // let b0 = self.sos[[s, 0]];
                        // let b1 = self.sos[[s, 1]];
                        // let b2 = self.sos[[s, 2]];
                        // // a0 = self.sos[[s, 3]] is typically 1.0
                        // let a1 = self.sos[[s, 4]];
                        // let a2 = self.sos[[s, 5]];
                        
                        // // Load current state
                        // let z1 = self.zi[[s, 0]];
                        // let z2 = self.zi[[s, 1]];
                        
                        // // // Direct Form II Transposed
                        // let y = x * b0 + z1;
                        let y = x.mul_add(sos[[s,0]],zi[[s,0]]);
                        // Update state (order matters!)
                        // self.zi[[s, 0]] = x * b1 + z2 - a1 * y;
                        // self.zi[[s, 1]] = x * b2 - a2 * y;
                        zi[[s, 0]] = x.mul_add(self.sos[[s,1]],y.mul_add(-self.sos[[s,4]],zi[[s,1]]));
                        zi[[s,1]] = x.mul_add(self.sos[[s,2]],-self.sos[[s,5]]*y);
                        // Output of this section becomes input to next section
                        x = y;
                    }
                    
                    signal[n] = x; // Final output after all sections
                }
            }
        }
        pub fn filter_inplace(signal: &mut ArrayViewMut1<f32>, sos: ArrayView2<f32>) {
            let n_samples = signal.len();
            let n_sections = sos.nrows();

            let mut z1 = vec![0.0f32; n_sections];
            let mut z2 = vec![0.0f32; n_sections];

            for n in 0..n_samples {
                let mut x = signal[n];

                for s in 0..n_sections {
                    let b0 = sos[[s, 0]];
                    let b1 = sos[[s, 1]];
                    let b2 = sos[[s, 2]];
                    let a1 = sos[[s, 4]];
                    let a2 = sos[[s, 5]];

                    let y = b0.mul_add(x, z1[s]);
                    z1[s] = b1.mul_add(x, -a1 * y + z2[s]);
                    z2[s] = b2.mul_add(x, -a2 * y);
                    x = y;
                }

                signal[n] = x;
            }
        }
        #[inline(always)]
        pub fn filter_unrolled_4_sections(signal: &mut ArrayViewMut1<f32>, sos: ArrayView2<f32>) {
            let b0_0 = sos[[0, 0]]; let b1_0 = sos[[0, 1]]; let b2_0 = sos[[0, 2]];
            let a1_0 = sos[[0, 4]]; let a2_0 = sos[[0, 5]];
            let b0_1 = sos[[1, 0]]; let b1_1 = sos[[1, 1]]; let b2_1 = sos[[1, 2]];
            let a1_1 = sos[[1, 4]]; let a2_1 = sos[[1, 5]];
            let b0_2 = sos[[2, 0]]; let b1_2 = sos[[2, 1]]; let b2_2 = sos[[2, 2]];
            let a1_2 = sos[[2, 4]]; let a2_2 = sos[[2, 5]];
            let b0_3 = sos[[3, 0]]; let b1_3 = sos[[3, 1]]; let b2_3 = sos[[3, 2]];
            let a1_3 = sos[[3, 4]]; let a2_3 = sos[[3, 5]];
    
            let mut z1_0 = 0.0f32; let mut z2_0 = 0.0f32;
            let mut z1_1 = 0.0f32; let mut z2_1 = 0.0f32;
            let mut z1_2 = 0.0f32; let mut z2_2 = 0.0f32;
            let mut z1_3 = 0.0f32; let mut z2_3 = 0.0f32;
            
            for x in signal.iter_mut() {
                let input = *x;
                // y = b0*x*z1
                // z1 = b1*x-a1*y+z2
                //z2 = b2*x-a2*y
                //x=y
                // Section 0
                let y0 = b0_0.mul_add(input, z1_0);
                z1_0 = b1_0.mul_add(input, a1_0.mul_add(-y0, z2_0));
                z2_0 = b2_0.mul_add(input, -a2_0 * y0);
                
                // Section 1  
                let y1 = b0_1.mul_add(y0, z1_1);
                z1_1 = b1_1.mul_add(y0, a1_1.mul_add(-y1, z2_1));
                z2_1 = b2_1.mul_add(y0, -a2_1 * y1);
                //Section 3
                let y2 = b0_2.mul_add(y1, z1_2);
                z1_2 = b1_2.mul_add(y1, a1_2.mul_add(-y2, z2_2));
                z2_2 = b2_2.mul_add(y1, -a2_2 * y2);
                // Section 4
                let y3 = b0_3.mul_add(y2, z1_3);
                z1_3 = b1_3.mul_add(y2, a1_3.mul_add(-y3, z2_3));
                z2_3 = b2_3.mul_add(y2, -a2_3 * y3);
                
                *x = y3;
            }
        }

        
        
        use ndarray::{Axis,ArrayViewMut3};
        
        use super::filters::butter_bandpass;
       // Method 2: Using ndarray's parallel axis iteration (safer)
    
    

    #[inline]
    pub fn bandpass_swapped(stack: &mut ArrayViewMut3<f32>,dt: f32, f0: f32, f1: f32){
        
        // dimensions are (y,x,t) this time
        let fs = 1f32 / dt;
        
        let cutoffs = vec![f0,f1];
        let mut slices = unravel_front_axes_as_views(stack);
        
        let sos = butter_bandpass(4,  &cutoffs,fs);
        slices.par_iter_mut()
        .for_each(|slice|{
            filter_unrolled_4_sections(slice, sos.view());
        })
    }
    pub fn unravel_front_axes_as_views<'a>(data: &'a mut ArrayViewMut3<f32>)
        -> Vec<ArrayViewMut1<'a, f32>>
    {
        // each 'lane' along axis 2 is an ArrayViewMut1
        data.lanes_mut(Axis(2))
            .into_iter()
            .collect()
    }

    }
    }
    
    pub mod goertzel{
       
        use ndarray::{ArrayView1,Array2,Axis,ArrayView3};
        use std::f32::consts::PI;
        use rayon::prelude::*;
        
        #[inline(always)]
        /// Implement a Goerzel algorithm iteratively over a given arrayview
        /// Takes in precomputed w and coeff
        pub fn goerzel(w:f32,coeff:f32,signal:ArrayView1<f32>)->f32{
            

            let mut prev1 = 0f32;
            let mut prev2 = 0f32;

            for i in 0..signal.len()-1{
                let s = signal[i] +coeff*prev1-prev2;
                prev2 = prev1;
                prev1 = s;

            }
            (prev1*prev1 +prev2*prev2 -(coeff*prev1*prev2)).sqrt()
        }
        pub fn power_at_freq(stack: &ArrayView3<f32>,freq:f32,sample_interval:f32)->Array2<f32>{
            let n = stack.dim().2 as f32;
            let index = (freq*n*sample_interval).round();
            let w = 2f32*PI*index/n;
            let coeff = 2f32*(w).cos();
            let (height,width,_) = stack.dim();
            let views = unravel_front_axes_as_views_static(stack);
            let result = views
            .into_par_iter()     
            .map(|view| {
            
            goerzel(w, coeff, view)*2f32/n
        }).collect();
        
        Array2::from_shape_vec((height,width), result).expect("Problem in reshaping")
    
            
        }
        pub fn unravel_front_axes_as_views_static<'a>(data: &'a  ArrayView3<f32>)
            -> Vec<ArrayView1<'a, f32>>
        {
            // each 'lane' along axis 2 is an ArrayViewMut1
            let out:Vec<ArrayView1<'a, f32>> =data.lanes(Axis(2))
                .into_iter()
                .collect();
            out
        }
    }
pub mod ac{
    use ndarray::{ArrayView3,ArrayViewMut3,Axis,Array1,ArrayView2,ShapeError,Array2,Array3};
    use rayon::prelude::*;
    /// Gets mean over the first 2 axes in parallel;
    #[inline(always)]
    fn find_spatial_mean(&stack: &ArrayView3<f32>) -> Result<Array1<f32>, ShapeError>{
        let result:Vec<f32> = stack.axis_iter(Axis(2)).into_par_iter().map(|frame| frame.mean().unwrap()).collect();
        Array1::from_shape_vec(result.len(), result)
    }
    #[inline(always)]
    fn subtract_DC(stack: &mut ArrayViewMut3<f32>, dc: &ArrayView2<f32>) {
        stack.axis_iter_mut(Axis(2)).into_par_iter().for_each(|mut frame| {
            frame -= dc;
        });
    }
    use crate::goertzel::{unravel_front_axes_as_views_static};
    use crate::utils::processing::{clamp_index,cubic_hermite_interp};
    use crate::utils::filters::butter_bandpass;
    use crate::utils::sos::filter_unrolled_4_sections;
    use crate::goertzel::goerzel;
    use std::f32::consts::PI;
    fn process_stack(&stack:&ArrayView3<f32>,
        frequency:f32, // to extract
        old_dt:f32, //current delta_t
        new_dt:f32, //target delta_t
        cutoffs:&[f32], //bandpass cutoff frequencies (absolute)
        interpolate:bool, //interpolate?
        filter:bool//filter?
    )->Result<(Array3<f32>,Array2<f32>),Box<dyn std::error::Error>>{
        
        let (height, width, ntimes) = stack.dim();
        let total_time = (ntimes - 1) as f32 * old_dt;
        let new_time_points = (total_time / new_dt).floor() as usize + 1;
        let scale_factor = (ntimes - 1) as f32 / (new_time_points - 1) as f32;
        let fs = if interpolate{1f32/new_dt}else{1f32/old_dt};
        let final_n = if interpolate{new_time_points} else {ntimes};
        let final_dt = if interpolate {new_dt} else {old_dt};
        let index = frequency*final_n as f32*final_dt;
        let w = 2f32*PI*index/final_n as f32;
        let coeff = 2f32*(w).cos();
        
    
        let interp_params = if interpolate{
            let interp_params: Vec<(usize, usize, usize, usize, f32)> = (0..new_time_points)
                .into_par_iter()
                .map(|i| {
                    let x = i as f32 * scale_factor;
                    let index = x as usize;
                    let t = x - x.floor();
                    let idx_a = clamp_index(index.saturating_sub(1), ntimes);
                    let idx_b = clamp_index(index, ntimes);
                    let idx_c = clamp_index(index + 1, ntimes);
                    let idx_d = clamp_index(index + 2, ntimes);
                    (idx_a, idx_b, idx_c, idx_d, t)
                })
                .collect();
            interp_params
        } else{vec![(0,0,0,0,0.0);5]};
        
        let sos = butter_bandpass(4,  cutoffs,fs);
        let slices = unravel_front_axes_as_views_static(&stack);
        
        let mut processed_series = Array3::zeros((height,width,final_n));
        let mut ac = Array2::zeros((height,width));
        let results: Vec<(usize, usize, Array1<f32>, f32)> = slices
                .par_iter()
                .enumerate()
                .map(|(idx, lane)| {
                    let mut signal =
                        if interpolate {
                            Array1::from_elem(new_time_points, 0.0)
                                .iter()
                                .enumerate()
                                .map(|(t, _)| {
                                    let (idx_a, idx_b, idx_c, idx_d, interp_t) = interp_params[t];
                                    cubic_hermite_interp(
                                        lane[idx_a],
                                        lane[idx_b],
                                        lane[idx_c],
                                        lane[idx_d],
                                        interp_t,
                                    )
                                })
                                .collect()
                        } else {
                            lane.to_owned()
                        };
                    signal -= signal.mean().unwrap();
                    if filter{filter_unrolled_4_sections(&mut signal.view_mut(), sos.view());}
                    
                    let n = signal.dim();
                    let power = goerzel(w, coeff, signal.view()) * 2f32 / n as f32;
                    
                    // CORRECT indexing for row-major order from lanes(Axis(2))
                    // lanes(Axis(2)) iterates as: (0,0), (0,1), ..., (0,width-1), (1,0), (1,1), ...
                    let y = idx / width;  // row index
                    let x = idx % width;  // column index
                    
                    (y, x, signal, power)
                })
                .collect();
       
        for (y, x, signal, power) in results {
            ac[[y, x]] = power;
            processed_series.slice_mut(s![ y, x,..]).assign(&signal);
        }
        
                         
        Ok((processed_series,ac))
    }
    
    ///Find optimal limits in (y,x,t) dimension stack
    /// by analysing discontinutities in signals
    #[inline]
    fn find_limits(stack:&ArrayView3<f32>,start:usize,end:usize,fs:f32,frequency:f32) -> Result<(usize, usize,usize), Box<dyn std::error::Error>> {
        
        let (_,_,n) = stack.dim();
        let mean_signal = find_spatial_mean(stack).expect("Getting mean went wrong");
        
        let std = mean_signal.std(0f32);
        // let time = Array1::range(0f32,n as f32*args.dt+args.dt/2f32,args.dt);
        let defaults = vec![start,end];
        let diff: Vec<f32> = (1..n).map(|i| mean_signal[[i]] - mean_signal[i - 1]).collect();
        let found_outliers:Vec<usize> = diff.iter().enumerate().filter_map(|(i, &x)| if x.abs() > 3f32 * std { Some(i) } else { None }).collect();
        let mut outliers = Vec::new();
        outliers.extend(defaults);
        outliers.extend(found_outliers);
        outliers.sort();
        let outlier_diffs:Vec<usize> = (1..outliers.len()).map(|i|outliers[i]-outliers[i-1]).collect();
        let max_idx = outlier_diffs
            .iter()
            .enumerate()
            .max_by_key(|&(_, value)| value)
            .map(|(idx, _)| idx).expect("Cannot find max index");
        let start = outliers[max_idx];
        let end = outliers[max_idx+1];
        let nperiods = ((end as f32-start as f32+1.0)*frequency/fs).floor() as usize;

        Ok((start, end, nperiods))  

    }
    use pyo3::prelude::*;
    use numpy::{PyArray, PyReadonlyArray3,ToPyArray};
     // <-- Add this import for the trait
    use ndarray::{s,Dim};

    

   
    use std::time::Instant;
    #[pyfunction]
    pub fn get_ac_data<'py>(
        py: Python<'py>,
        raws: PyReadonlyArray3<f32>,
        frequency: f32,
        framerate: f32,
        start: f32,
        end: f32,
        hardlimits: bool,
        interpolation: bool,
        filt: bool,
        periods: f32,
    ) -> PyResult<(
    Bound<'py, PyArray<f32, Dim<[usize; 2]>>>,
    Bound<'py, PyArray<f32, Dim<[usize; 2]>>>,
    (
        Bound<'py, PyArray<f32, Dim<[usize; 1]>>>,
        Bound<'py, PyArray<f32, Dim<[usize; 3]>>>
    ),
    (usize,usize),
)> {
        // Create read-only view
        let raws_view = raws.as_array();
        
        // Unpack or assign start, end, nperiods
        let time = Instant::now();
        let (start, end, nperiods) = if !hardlimits {
            find_limits(&raws_view, start as usize, end as usize, framerate, frequency).unwrap()
        } else {
            (start as usize, end as usize, periods as usize)
        };
        let elapsed = time.elapsed();
        println!("Time to find limits:{:?}",elapsed);
        // Adjust end to match whole number of periods
        println!("Current limits: {:?} {:?}", start, end);
        println!("Start: {}, End: {}, NPeriods: {}", start, end, nperiods);
        println!("Frequency: {}, Framerate: {}", frequency, framerate);
        let mut adjusted_nperiods = nperiods;
        let (final_start,final_end) = if nperiods <1{(start,end)}else{
        
            // Calculate new_last_index with rounding
            let mut new_last_index = ((adjusted_nperiods as f32 / frequency) * framerate).round() as isize;

            // Cast start and end to isize for safe arithmetic
            let start_isize = start as isize;
            let end_isize = end as isize;

            // Safety loop: ensure new_last_index + start does not exceed end
            while new_last_index + start_isize > end_isize && adjusted_nperiods > 0 {
                adjusted_nperiods -= 1;
                new_last_index = ((adjusted_nperiods as f32 / frequency) * framerate).round() as isize;
            }

            // Handle case where no suitable period fits
            if adjusted_nperiods == 0 && new_last_index + start_isize > end_isize {
                println!("Warning: Could not find suitable period fitting within limits.");
            }

            // Now compute final indices clamping within raws_view bounds

            // raws_view.dim() returns (dim0, dim1, dim2) tuple, confirm dim2 is the correct time dimension
            let max_index = (raws_view.dim().2 as isize) - 1;

            // Clamp new_last_index so final end does not exceed max_index
            if new_last_index + start_isize > max_index {
                new_last_index = max_index - start_isize;
            }
            // Clamp start (should never be negative but just in case)
            let start_adj = if start_isize < 0 { 0 } else { start_isize } as usize;
            let end_adj = (start_isize + new_last_index) as usize;
            println!("{:?},{:?}",start_adj,end_adj);
            (start_adj,end_adj)
        };

        
        //Slice inputs
        let  stack = raws_view.slice(s![.., .., final_start..=final_end]);

        // Compute DC component
        let time = Instant::now();
        let dc = stack.mean_axis(Axis(2)).unwrap();
        let elapsed = time.elapsed();
        println!("Time to compute mean:{:?}",elapsed);
        let framerate = if interpolation {framerate.ceil()} else {framerate};
        let (stack,ac) = process_stack(&stack, 
                                            frequency, 
                                            1f32/framerate.round(),
                                            1f32/framerate,
                                            &[0.1,6.0],
                                            interpolation,filt)
                                            .expect("Failed to process stack");
        // Create time axis
        let num_samples = stack.dim().2;
        println!("Framerate is {:?}",framerate);
        println!("Stack size is{:?}",num_samples);
        let times = Array1::from_vec((0..num_samples).map(|x| x as f32 / framerate).collect());
        println!("Made time!");
        // Convert results to Python/NumPy
        let time = Instant::now();
        let ac_py = ac.to_pyarray(py);
        let dc_py = dc.to_pyarray(py);
        let time_py = times.to_pyarray(py);
        let stack_py = stack.to_pyarray(py);
        // → shape [X, Y, new_Z]         // → shape [new_Z, X, Y]                        // → ensure it's contiguous
     
         let elapsed = time.elapsed();
        println!("Time to prepare results:{:?}",elapsed);
        
        Ok((ac_py, dc_py, (time_py,stack_py),(final_start,final_end)))
    }

}
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn _ac_processor(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::ac::get_ac_data, m)?)?;  
    Ok(())
}
#[cfg(test)]
mod tests {

}
