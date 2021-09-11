data {
   int N_batch; //number of batches
   int N; //no. of samples
   int batch[N]; //batch of considered sample
}

transformed data {

   real reset_intercept_sim;
   real<lower=0> shape_sim;

   real mu_batch_sim;             // Location intercepts population mean
   real<lower=0> sigma_batch_sim; // Location intercepts population standard deviation
   vector[N_batch] alpha_batch_tilde_sim; // Non-centered batch intercepts
   vector[N_batch] batch_coef_sim; 
   int sim_resets[N];
   real sim_failure_time[N];
   
   mu_batch_sim = normal_rng(0, 1);           // Prior model
   sigma_batch_sim = fabs(normal_rng(0, 2));       // Prior model
   for (k in 1:N_batch){
       alpha_batch_tilde_sim[k] = normal_rng(0,1);// Non-centered hierarchical model
   }
   reset_intercept_sim = normal_rng(3,1);
   shape_sim = fabs(normal_rng(8,1));
   batch_coef_sim = mu_batch_sim + sigma_batch_sim * alpha_batch_tilde_sim;

    for (n in 1:N) {
        sim_resets[n] = poisson_rng(exp(reset_intercept_sim+batch_coef_sim[batch[n]]));
        sim_failure_time[n] = gamma_rng(shape_sim,exp(batch_coef_sim[batch[n]]));
    }

}

parameters {
   real reset_intercept;
   real<lower=0> shape;

   real mu_batch;             // Location intercepts population mean
   real<lower=0> sigma_batch; // Location intercepts population standard deviation
   vector[N_batch] alpha_batch_tilde; // Non-centered batch intercepts
 }


transformed parameters {
   /* ... declarations ... statements ... */
   vector[N_batch] batch_coef = mu_batch + sigma_batch * alpha_batch_tilde;
}

model {
   /* ... declarations ... statements ... */
   mu_batch ~ normal(0, 1);           // Prior model
   sigma_batch ~ normal(0, 2);       // Prior model
   alpha_batch_tilde ~ normal(0, 1); // Non-centered hierarchical model
   reset_intercept ~ normal(3,1);
   shape ~ normal(8,1);
   sim_resets ~ poisson(exp(reset_intercept+batch_coef[batch]));
   sim_failure_time ~ gamma(shape, exp(batch_coef[batch]));  
}


generated quantities {
    int<lower = 0, upper = 1> lt_sim[8]
      = { reset_intercept < reset_intercept_sim, 
          shape < shape_sim,
          mu_batch < mu_batch_sim,
          sigma_batch < sigma_batch_sim,
          alpha_batch_tilde[1] <alpha_batch_tilde_sim[1],
          alpha_batch_tilde[2] <alpha_batch_tilde_sim[2],
          alpha_batch_tilde[3] <alpha_batch_tilde_sim[3],
          alpha_batch_tilde[4] <alpha_batch_tilde_sim[4]};
}