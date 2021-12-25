data {
   int N; //no. of samples
   int resets[N]; //no. of resets in observed period
   real failure_time[N]; //no. of failures in observed period
}

parameters {
   real reset_intercept;
   real<lower=0> shape;
   real type_coef;
   real<lower=0> inv_phi; // Over-dispersion parameter

}


transformed parameters {
  // Save phi
  real<lower=0> phi = 1 / inv_phi;
}

model {
   type_coef ~ normal(0,3);
   shape ~ normal(8,1);
   reset_intercept ~ normal(3,1);
   resets ~ poisson(exp(reset_intercept+type_coef));
   resets ~ neg_binomial_2_log(reset_intercept+type_coef, phi);
  
}

generated quantities {
    int pred_resets[N];
    real pred_failure_time[N];
    for (n in 1:N) {
        pred_resets[n] = neg_binomial_2_log_rng(reset_intercept+type_coef, phi);
        pred_failure_time[n] = gamma_rng(shape,exp(type_coef));
    }
}