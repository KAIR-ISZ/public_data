data {
   int N_batch; //number of batches
   int N; //number of samples to generate
}

transformed data {
   real reset_intercept = 3;
   real reset_slope = 1;
   real<lower=0> shape = 8;
   real failure_intercept = 0;
   real failure_slope =1;
   vector[N_batch] theta = uniform_simplex(N_batch);
}

generated quantities {

   int<lower=0,upper=N_batch> batch[N];
   real batch_coef[N_batch]; 
   int<lower=0> resets[N];
   real<lower=0> failure_time[N];
   for (k in 1:N_batch){
       batch_coef[k] = normal_rng(1,1);
   }
   for (n in 1:N) {
        batch[n] = categorical_rng(theta);
        resets[n] = poisson_rng(exp(reset_intercept+reset_slope*batch_coef[batch[n]]));
        failure_time[n] = gamma_rng(shape,exp(failure_intercept+failure_slope*batch_coef[batch[n]]));
    }
}