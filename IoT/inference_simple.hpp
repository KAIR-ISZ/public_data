
// Code generated by stanc v2.29.1
#include <stan/model/model_header.hpp>
namespace inference_simple_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 22> locations_array__ = 
{" (found before start of program)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 8, column 3 to column 24)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 9, column 3 to column 23)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 10, column 3 to column 18)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 22, column 4 to column 23)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 23, column 4 to column 30)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 25, column 8 to column 69)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 26, column 8 to column 63)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 24, column 19 to line 27, column 5)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 24, column 4 to line 27, column 5)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 14, column 3 to column 33)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 15, column 3 to column 27)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 16, column 3 to column 23)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 17, column 3 to column 52)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 18, column 3 to column 47)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 2, column 3 to column 9)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 3, column 14 to column 15)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 3, column 3 to column 17)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 4, column 21 to column 22)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 4, column 3 to column 24)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 22, column 20 to column 21)",
 " (in '/Users/jerzybaranowski/GitHub/KAIR-ISZ/public_data/IoT/inference_simple.stan', line 23, column 27 to column 28)"};




class inference_simple_model final : public model_base_crtp<inference_simple_model> {

 private:
  int N;
  std::vector<int> resets;
  std::vector<double> failure_time; 
  
 
 public:
  ~inference_simple_model() { }
  
  inline std::string model_name() const final { return "inference_simple_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.29.1", "stancflags = "};
  }
  
  
  inference_simple_model(stan::io::var_context& context__,
                         unsigned int random_seed__ = 0,
                         std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "inference_simple_model_namespace::inference_simple_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 15;
      context__.validate_dims("data initialization","N","int",
           std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      
      
      current_statement__ = 15;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 16;
      stan::math::validate_non_negative_index("resets", "N", N);
      current_statement__ = 17;
      context__.validate_dims("data initialization","resets","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      resets = std::vector<int>(N, std::numeric_limits<int>::min());
      
      
      current_statement__ = 17;
      resets = context__.vals_i("resets");
      current_statement__ = 18;
      stan::math::validate_non_negative_index("failure_time", "N", N);
      current_statement__ = 19;
      context__.validate_dims("data initialization","failure_time","double",
           std::vector<size_t>{static_cast<size_t>(N)});
      failure_time = 
        std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 19;
      failure_time = context__.vals_r("failure_time");
      current_statement__ = 20;
      stan::math::validate_non_negative_index("pred_resets", "N", N);
      current_statement__ = 21;
      stan::math::validate_non_negative_index("pred_failure_time", "N", N);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1 + 1;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "inference_simple_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      local_scalar_t__ reset_intercept = DUMMY_VAR__;
      current_statement__ = 1;
      reset_intercept = in__.template read<local_scalar_t__>();
      local_scalar_t__ shape = DUMMY_VAR__;
      current_statement__ = 2;
      shape = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                0, lp__);
      local_scalar_t__ type_coef = DUMMY_VAR__;
      current_statement__ = 3;
      type_coef = in__.template read<local_scalar_t__>();
      {
        current_statement__ = 10;
        lp_accum__.add(
          stan::math::normal_lpdf<propto__>(reset_intercept, 3, 1));
        current_statement__ = 11;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(type_coef, 0, 3));
        current_statement__ = 12;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(shape, 8, 1));
        current_statement__ = 13;
        lp_accum__.add(
          stan::math::poisson_lpmf<propto__>(resets,
            stan::math::exp((reset_intercept + type_coef))));
        current_statement__ = 14;
        lp_accum__.add(
          stan::math::gamma_lpdf<propto__>(failure_time, shape,
            stan::math::exp(type_coef)));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "inference_simple_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      double reset_intercept = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      reset_intercept = in__.template read<local_scalar_t__>();
      double shape = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      shape = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(
                0, lp__);
      double type_coef = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      type_coef = in__.template read<local_scalar_t__>();
      out__.write(reset_intercept);
      out__.write(shape);
      out__.write(type_coef);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      std::vector<int> pred_resets =
         std::vector<int>(N, std::numeric_limits<int>::min());
      std::vector<double> pred_failure_time =
         std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 9;
      for (int n = 1; n <= N; ++n) {
        current_statement__ = 6;
        stan::model::assign(pred_resets,
          stan::math::poisson_rng(
            stan::math::exp((reset_intercept + type_coef)), base_rng__),
          "assigning variable pred_resets", stan::model::index_uni(n));
        current_statement__ = 7;
        stan::model::assign(pred_failure_time,
          stan::math::gamma_rng(shape, stan::math::exp(type_coef),
            base_rng__),
          "assigning variable pred_failure_time", stan::model::index_uni(n));
      }
      out__.write(pred_resets);
      out__.write(pred_failure_time);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_std_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ reset_intercept = DUMMY_VAR__;
      reset_intercept = in__.read<local_scalar_t__>();
      out__.write(reset_intercept);
      local_scalar_t__ shape = DUMMY_VAR__;
      shape = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, shape);
      local_scalar_t__ type_coef = DUMMY_VAR__;
      type_coef = in__.read<local_scalar_t__>();
      out__.write(type_coef);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"reset_intercept", "shape",
      "type_coef", "pred_resets", "pred_failure_time"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
      std::vector<size_t>{}, std::vector<size_t>{},
      std::vector<size_t>{static_cast<size_t>(N)},
      std::vector<size_t>{static_cast<size_t>(N)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "reset_intercept");
    param_names__.emplace_back(std::string() + "shape");
    param_names__.emplace_back(std::string() + "type_coef");
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "pred_resets" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "pred_failure_time" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "reset_intercept");
    param_names__.emplace_back(std::string() + "shape");
    param_names__.emplace_back(std::string() + "type_coef");
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "pred_resets" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "pred_failure_time" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"reset_intercept\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"shape\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"type_coef\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"pred_resets\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"int\"}},\"block\":\"generated_quantities\"},{\"name\":\"pred_failure_time\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"reset_intercept\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"shape\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"type_coef\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"pred_resets\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"int\"}},\"block\":\"generated_quantities\"},{\"name\":\"pred_failure_time\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  ((1 + 1) + 1);
      const size_t num_transformed = 0;
      const size_t num_gen_quantities = 
  (N + N);
      std::vector<double> vars_vec(num_params__
       + (emit_transformed_parameters * num_transformed)
       + (emit_generated_quantities * num_gen_quantities));
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        vars_vec.data(), vars_vec.size());
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  ((1 + 1) + 1);
      const size_t num_transformed = 0;
      const size_t num_gen_quantities = 
  (N + N);
      vars.resize(num_params__
        + (emit_transformed_parameters * num_transformed)
        + (emit_generated_quantities * num_gen_quantities));
      write_array_impl(base_rng, params_r, params_i, vars, emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 3> names__{"reset_intercept", "shape",
      "type_coef"};
      const std::array<Eigen::Index, 3> constrain_param_sizes__{1, 1, 1};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}
using stan_model = inference_simple_model_namespace::inference_simple_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return inference_simple_model_namespace::profiles__;
}

#endif

