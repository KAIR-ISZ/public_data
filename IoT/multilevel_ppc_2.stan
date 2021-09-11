data {
   int N_batch; //number of batches
   int N; //no. of samples
   int batch[N]; //batch of considered sample
}


generated quantities {
   real reset_intercept;
   real<lower=0> shape;

   real mu_batch;             // Location intercepts population mean
   real<lower=0> sigma_batch; // Location intercepts population standard deviation
   vector[N_batch] alpha_batch_tilde; // Non-centered batch intercepts
   vector[N_batch] batch_coef; 
   int pred_resets[N];
   real pred_failure_time[N];
   
   mu_batch = normal_rng(0, 1);           // Prior model
   sigma_batch = fabs(normal_rng(0, 2));       // Prior model
   for (k in 1:N_batch){
       alpha_batch_tilde[k] = normal_rng(0,1);// Non-centered hierarchical model
   }
   reset_intercept = normal_rng(3,1);
   shape = fabs(normal_rng(8,1));
   batch_coef = mu_batch + sigma_batch * alpha_batch_tilde;

   /* ... declarations ... statements ... */
    for (n in 1:N) {
        pred_resets[n] = poisson_rng(exp(reset_intercept+batch_coef[batch[n]]));
        pred_failure_time[n] = gamma_rng(shape,exp(batch_coef[batch[n]]));
    }
}