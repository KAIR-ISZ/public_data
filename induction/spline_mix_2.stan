data {
  int<lower=1> K;   // number of predictors
  int<lower=1> M;  // number of mixture components
  simplex [M] lambda0; // prior for mixture components

  int<lower=1> I;   // number of data items
  array[I] int<lower=1> N;   // number of samples per item
  int<lower=1> total_len;
  row_vector[total_len*K] x;   // predictor matrix
  row_vector[total_len] y;      // matrix of outputs

  int<lower=1> IL;   // number of labeled data items
  array[IL] int<lower=1> N_labeled;   // number of samples per item
  int<lower=1> total_len_labeled;
  row_vector[total_len_labeled*K] x_labeled;   // predictor matrix
  row_vector[total_len_labeled] y_labeled;      // matrix of labeled outputs
  array[IL] int<lower=1,upper=M> labels;

  int<lower=1> IT;   // number of test data items
  array[IT] int<lower=1> N_test;   // number of samples per item
  int<lower=1> total_len_test ;
  row_vector[total_len_test] y_test;      // matrix of test outputs
  row_vector[total_len_test*K] x_test;   // predictor matrix
}

parameters {
  matrix[K,M] beta;      // coefficients on Q_ast for each mixture component
  array [M] real<lower=0> sigma;  // error scale for each mixture component
  simplex [M] lambda; //mixture components
  real alpha;
}

model {
  alpha ~ exponential(1);
  for (m in 1:M) {
   /* code */
   target += std_normal_lpdf(beta[1:K,m]);
  }

  vector[total_len_labeled] mu_labeled ;
  int current = 1;
  int current_x = 1;
  for (n in 1:IL){
    for (z in 1:N_labeled[n]){
    mu_labeled[current] = x_labeled[current_x : current_x + K -1] * beta[1:K,labels[n]] + alpha;
    current = current + 1;
    current_x = current_x + K;
    }
  }

  matrix[total_len,M] mu;
  current = 1;
  current_x = 1;
  for (n in 1:I){
    for ( z in 1:N[n]){
      for (m in 1:M){
        mu[current,m] = x[current_x : current_x + K -1] * beta[1:K,m] + alpha;
      }
      current = current + 1;
      current_x = current_x + K;
    }
  }
  
  vector[M] log_lambda = log(lambda);  // cache log calculation



  sigma ~ exponential(.1);
  lambda ~ dirichlet(lambda0);
  
  current = 1;
  for (n in 1:IL) {
    target += normal_lpdf(y_labeled[current:current+N_labeled[n]-1] | mu_labeled[current:current+N_labeled[n]-1], sigma[labels[n]]);
    current = current + N_labeled[n];
  }  


  current = 1;
  for (n in 1:I) {
    vector[M] lps = log_lambda;
    for (m in 1:M) {
      lps[m] += normal_lpdf(y[current:current+N[n]-1] | mu[current:current+N[n]-1,m], sigma[m]);
    }
    target += log_sum_exp(lps);
    current = current + N[n];
  }

}
generated quantities {
  matrix[M,IT] probabilities;
  matrix[M,IT] log_probabilities;
  
  matrix[total_len_test,M] mu_test;
  int current = 1;
  int current_x = 1;

  for (n in 1:IT){
    for (z in 1:N_test[n]){
      for (m in 1:M){
        mu_test[current,m] = x_test[current_x : current_x + K -1] * beta[1:K,m] + alpha;
      }
      current = current + 1;
      current_x = current_x + K;
    }
  }


  real normalizer;
  current = 1;
   for (n in 1:IT) {
      /* code */
      for (m in 1:M) {
         /* code */
          log_probabilities[m,n]=normal_lpdf(y_test[current:current+N_test[n]-1]| mu_test[current:current+N_test[n]-1,m], sigma[m]) +log(lambda[m]);   
      }
      normalizer = log_sum_exp( log_probabilities[1:M,n]);
      log_probabilities[1:M,n] = log_probabilities[1:M,n]-normalizer;
      current = current + N_test[n];
   }
  
  probabilities = exp(log_probabilities);
  
}
