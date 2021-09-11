data {
   int N; //no. of samples
   int resets[N]; //no. of resets in observed period
   real failure_time[N]; //no. of failures in observed period
}

parameters {
   real reset_intercept;
   real<lower=0> shape;
   real type_coef;
}

model {
   type_coef ~ normal(0,3);
   shape ~ normal(8,1);
   resets ~ poisson(exp(reset_intercept+type_coef));
   failure_time ~ gamma(shape, exp(type_coef));  
}

generated quantities {
    int pred_resets[N];
    real pred_failure_time[N];
    for (n in 1:N) {
        pred_resets[n] = poisson_rng(exp(reset_intercept+type_coef));
        pred_failure_time[n] = gamma_rng(shape,exp(type_coef));
    }
}