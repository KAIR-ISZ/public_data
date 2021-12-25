data {
   int N; //no. of samples
   int resets[N]; //no. of resets in observed period
   real failure_time[N]; //no. of failures in observed period
}

parameters {
   real<lower=0> reset_const;
   real<lower=0> shape;
   real<lower=0> failure_const;
   real mu_type;
   real<lower=0> sigma_type;
   real<lower=0> type_coef[N];
   }



model {
   mu_type ~ normal(0,2);
   sigma_type ~ normal(0,2);
   type_coef ~ normal(mu_type,sigma_type);
   for (n in 1:N) {
      resets[n] ~ poisson(reset_const*type_coef[N]);
      failure_time[n] ~ gamma(shape, failure_const*type_coef[n]);
   }
}