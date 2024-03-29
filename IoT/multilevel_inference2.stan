data {
   int N_batch; //number of batches
   int N; //no. of samples
   int batch[N]; //batch of considered sample
   int resets[N]; //no. of resets in observed period
   real failure_time[N]; //no. of failures in observed period
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
   resets ~ poisson(exp(reset_intercept+batch_coef[batch]));
   failure_time ~ gamma(shape, exp(batch_coef[batch]));  
}

generated quantities {
    int pred_resets[N];
    real pred_failure_time[N];
   /* ... declarations ... statements ... */
    for (n in 1:N) {
        pred_resets[n] = poisson_rng(exp(reset_intercept+batch_coef[batch[n]]));
        pred_failure_time[n] = gamma_rng(shape,exp(batch_coef[batch[n]]));
    }
}