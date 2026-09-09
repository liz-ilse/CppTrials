// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]

#include <RcppEigen.h>
#include <RcppNumerical.h>

// for multivariate integration
using namespace Numer;

// importing only the Rcpp types we will need
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::List;
using Rcpp::IntegerVector;

////////////////////////// Mean response function /////////////////////////////

// Internal scalar version
double mu_SR_scalar(double a_0, double a_1, double a_2, double a_3, double d)
{
  return a_0 + a_3 / (1.0 + std::exp(-a_1 * d + a_2));
} 

// Internal vector version
NumericVector mu_SR_internal(double a_0, double a_1, double a_2, double a_3,
                             NumericVector d)
{ 
  int n = d.size();
  NumericVector out(n);
  for (int i = 0; i < n; i++) {
    out[i] = mu_SR_scalar(a_0, a_1, a_2, a_3, d[i]);
  } 
  return out;
} 

// [[Rcpp::export]]
NumericVector mu_SR(double a_0, double a_1, double a_2, double a_3,
                    NumericVector d)
{ 
  return mu_SR_internal(a_0, a_1, a_2, a_3, d);
} 

////////////////////////// joint_prob code ////////////////////////////////////
// internal only - normalise joint pi_jk to sum to 1
// missing mass is redistributed to all edge cells weighted by their current probabilities
void normalise_joint_internal(NumericMatrix pi_cur, IntegerVector K)
{
  int K_T = K[0];
  int K_R = K[1];
   
  // current total probability
  double S = 0.0;
  for (int i = 0; i < K_T; i++) {
    for (int j = 0; j < K_R; j++) {
      S += pi_cur(i, j);
    } 
  }
  double missing = 1.0 - S;
   
  // sum of edge cells (outer ring) - first/last row and first/last column
  // taking care not to double-count the four corners
  double edge_sum = 0.0;
   
  // first row
  for (int j = 0; j < K_R; j++) edge_sum += pi_cur(0, j);
  // last row
  for (int j = 0; j < K_R; j++) edge_sum += pi_cur(K_T - 1, j);
  // first column excluding corners
  for (int i = 1; i < K_T - 1; i++) edge_sum += pi_cur(i, 0);
  // last column excluding corners
  for (int i = 1; i < K_T - 1; i++) edge_sum += pi_cur(i, K_R - 1);
   
  // safeguard: if all edge cells are zero we cannot proportionally
  // redistribute - return without modifying (extremely unlikely case)
  if (edge_sum <= 0.0) return;
   
  // redistribute missing mass to edge cells weighted by current probability
  // first row
  for (int j = 0; j < K_R; j++) {
    pi_cur(0, j) += missing * pi_cur(0, j) / edge_sum;
  } 
  // last row
  for (int j = 0; j < K_R; j++) {
    pi_cur(K_T - 1, j) += missing * pi_cur(K_T - 1, j) / edge_sum;
  } 
  // first column excluding corners
  for (int i = 1; i < K_T - 1; i++) {
    pi_cur(i, 0) += missing * pi_cur(i, 0) / edge_sum;
  } 
  // last column excluding corners
  for (int i = 1; i < K_T - 1; i++) {
    pi_cur(i, K_R - 1) += missing * pi_cur(i, K_R - 1) / edge_sum;
  }
} 

// helper to cap infinite bounds since Rcpp integrate has difficulty with infinite values
double cap_bound(double x, double mu, double sig, double n_sd = 6.0) {
  if (x == R_PosInf) return mu + n_sd * sig;
  if (x == R_NegInf) return mu - n_sd * sig;
  return x;
}

class BivariateNormal : public MFunc
{
private: 
  const double mu1, mu2;
  double const1;  // 2 * (1 - rho^2)
  double const2;  // 1 / (2 * PI * sig1^2 * sqrt(1 - rho^2))
  double rho, sig1;
  
public: 
  BivariateNormal(const double& mu1_, const double& mu2_,
                  const double& sig1_, const double& rho_) :
  mu1(mu1_), mu2(mu2_), rho(rho_), sig1(sig1_)
  {
    const1 = 2.0 * (1.0 - rho * rho);
    const2 = 1.0 / (2.0 * M_PI * sig1 * sig1 * std::sqrt(1.0 - rho * rho));
  } 
  
  double operator()(Constvec& x)
  {
    double z1 = (x[0] - mu1) / sig1;
    double z2 = (x[1] - mu2) / sig1;
    double z  = z1*z1 - 2.0*rho*z1*z2 + z2*z2;
    return const2 * std::exp(-z / const1);
  }
}; 

// add a static counter to track failures
static int integration_failures = 0;

NumericMatrix joint_prob_internal(
    List t_bound, List r_bound,
    double mu_T, double mu_R,
    double sig1, double rho,
    IntegerVector K)
{
  BivariateNormal f(mu_T, mu_R, sig1, rho);
   
  double err_est;
  int err_code;
   
  NumericMatrix pi_cur(K[0], K[1]);
   
  for (int i = 0; i < K[0]; i++) {
    for (int j = 0; j < K[1]; j++) {
       
      NumericVector ti = t_bound[i];
      NumericVector rj = r_bound[j];
       
      Eigen::VectorXd lower(2), upper(2);
      lower << cap_bound(ti[0], mu_T, sig1), cap_bound(rj[0], mu_R, sig1);
      upper << cap_bound(ti[1], mu_T, sig1), cap_bound(rj[1], mu_R, sig1);
       
      pi_cur(i, j) = integrate(f, lower, upper, err_est, err_code, 
             10000,    // maxeval - increase from 1000
             1e-9,     // eps_abs
             1e-9);    // eps_rel);
       
      // silently count failures rather than printing each one
      if (err_code != 0) integration_failures++;
    } 
  }
  
  // clip any negative values from cubature error to zero
  // (cubature can produce small negatives when true probability is near zero)
  for (int i = 0; i < K[0]; i++) {
    for (int j = 0; j < K[1]; j++) {
      if (pi_cur(i, j) < 0.0) pi_cur(i, j) = 0.0;
    }
  }
  
  // normalize probability function
  normalise_joint_internal(pi_cur, K);
  
  return pi_cur;
} 

// [[Rcpp::export]]
int get_integration_failures() {
  return integration_failures;
} 

// [[Rcpp::export]]
void reset_integration_failures() {
  integration_failures = 0;
} 

// [[Rcpp::export]]
NumericMatrix joint_prob(
    List t_bound, List r_bound,
    double mu_T, double mu_R,
    double sig1, double rho,
    IntegerVector K)
{
  return joint_prob_internal(t_bound, r_bound, mu_T, mu_R, sig1, rho, K);
}


////////////////////////////// pi_jk_fxn code /////////////////////////////////

// Internal implementation
List pi_jk_fxn_internal(
    NumericVector e_T, NumericVector e_R,
    double a_0, double a_1, double a_2, double a_3,
    double b_0, double b_1,
    double sig1, double rho,
    NumericVector std_d,
    int n_doses,
    IntegerVector K)
{ 
  // bounds
  List t_bound(K[0]);
  List r_bound(K[1]);
   
  for (int k = 0; k < K[0]; k++) {
    t_bound[k] = NumericVector::create(e_T[k], e_T[k + 1]);
  }
   
  for (int k = 0; k < K[1]; k++) {
    r_bound[k] = NumericVector::create(e_R[k], e_R[k + 1]);
  }
   
  // means for each dose level
  NumericVector mu_R_list(n_doses);
  NumericVector mu_T_list(n_doses);
   
  for (int l = 0; l < n_doses; l++) {
    mu_R_list[l] = mu_SR_scalar(a_0, a_1, a_2, a_3, std_d[l]);
    mu_T_list[l] = b_0 + b_1 * std_d[l];
  }
   
  // joint probabilities for each dose level
  List pi_jk(n_doses);
   
  for (int l = 0; l < n_doses; l++) {
     
    NumericMatrix mat = joint_prob_internal(
      t_bound, r_bound, mu_T_list[l], mu_R_list[l], sig1, rho, K);
     
    // set row and column names
    Rcpp::CharacterVector rnames(K[0]);
    Rcpp::CharacterVector cnames(K[1]);
    for (int i = 0; i < K[0]; i++) rnames[i] = std::to_string(i);
    for (int j = 0; j < K[1]; j++) cnames[j] = std::to_string(j);
     
    Rcpp::rownames(mat) = rnames;
    Rcpp::colnames(mat) = cnames;
     
    pi_jk[l] = mat;
  } 
  
  return pi_jk;
} 

// [[Rcpp::export]]
List pi_jk_fxn(
    NumericVector e_T, NumericVector e_R,
    double a_0, double a_1, double a_2, double a_3,
    double b_0, double b_1,
    double sig1, double rho,
    NumericVector std_d,
    int n_doses,
    IntegerVector K)
{ 
  return pi_jk_fxn_internal(e_T, e_R, a_0, a_1, a_2, a_3,
                            b_0, b_1, sig1, rho, std_d, n_doses, K); 
}

/////////////////////// posterior pi_jk function //////////////////////////////

// internal implementation
List post_pi_jk_fxn_internal(
    NumericVector cur_lamb_T, NumericVector cur_lamb_R,
    double cur_a_0, double cur_a_1, double cur_a_2, double cur_a_3,
    double cur_b_0, double cur_b_1,
    double sig1, double rho,
    NumericVector std_d,
    int n_doses,
    IntegerVector K)
{
  // toxicity cutpoints
  NumericVector c_o_T(K[0] - 1);
  c_o_T[0] = 0.0;
  for (int k = 0; k < K[0] - 2; k++) {
    c_o_T[k + 1] = cur_lamb_T[k] + c_o_T[k];
  }
   
  // e_T = c(-Inf, c_o_T, Inf)
  NumericVector e_T(K[0] + 1);
  e_T[0] = R_NegInf;
  for (int k = 0; k < K[0] - 1; k++) {
    e_T[k + 1] = c_o_T[k];
  } 
  e_T[K[0]] = R_PosInf;
   
  // response cutpoints
  NumericVector c_o_R(K[1] - 1);
  c_o_R[0] = 0.0;
  for (int k = 0; k < K[1] - 2; k++) {
    c_o_R[k + 1] = cur_lamb_R[k] + c_o_R[k];
  }
   
  // e_R = c(-Inf, c_o_R, Inf)
  NumericVector e_R(K[1] + 1);
  e_R[0] = R_NegInf;
  for (int k = 0; k < K[1] - 1; k++) {
    e_R[k + 1] = c_o_R[k];
  } 
  e_R[K[1]] = R_PosInf;
    
  return pi_jk_fxn_internal(e_T, e_R, cur_a_0, cur_a_1, cur_a_2, cur_a_3,
                            cur_b_0, cur_b_1, sig1, rho, std_d, n_doses, K);
} 

//////////////////////////////// z update ///////////////////////////////////
// internal only
NumericVector samp_z_internal(
    double rho,
    NumericVector lambda_T, NumericVector lambda_R,
    NumericVector z_R,
    NumericVector y_T_vec, NumericVector y_R_vec, NumericVector std_d_vec,
    NumericVector mu_T, NumericVector mu_R,
    double sig_2,
    int n_doses,
    IntegerVector K)
{
  int n = z_R.size();
   
  // sorted unique dose levels
  NumericVector dose_level = Rcpp::sort_unique(std_d_vec);
   
  double cur_var = sig_2 * (1.0 - rho * rho);
  double cur_sd  = std::sqrt(cur_var);
  
  // toxicity cutpoints
  NumericVector c_o_T(K[0] - 1);
  c_o_T[0] = 0.0;
  for (int k = 0; k < K[0] - 2; k++) {
    c_o_T[k + 1] = lambda_T[k] + c_o_T[k];
  }
  
  NumericVector e_T(K[0] + 1);
  e_T[0] = R_NegInf;
  for (int k = 0; k < K[0] - 1; k++) {
    e_T[k + 1] = c_o_T[k];
  } 
  e_T[K[0]] = R_PosInf;
   
  // response cutpoints
  NumericVector c_o_R(K[1] - 1);
  c_o_R[0] = 0.0;
  for (int k = 0; k < K[1] - 2; k++) {
    c_o_R[k + 1] = lambda_R[k] + c_o_R[k];
  } 
  
  NumericVector e_R(K[1] + 1);
  e_R[0] = R_NegInf;
  for (int k = 0; k < K[1] - 1; k++) {
    e_R[k + 1] = c_o_R[k];
  } 
  e_R[K[1]] = R_PosInf;
   
  // bounds
  List t_bound(K[0]);
  List r_bound(K[1]);
   
  for (int k = 0; k < K[0]; k++) {
    t_bound[k] = NumericVector::create(e_T[k], e_T[k + 1]);
  } 
  
  for (int k = 0; k < K[1]; k++) {
    r_bound[k] = NumericVector::create(e_R[k], e_R[k + 1]);
  } 
  
  // z_T update
  NumericVector new_z_T(n, NA_REAL);
   
  // iterate over observed unique doses only (dose_level.size() may be < n_doses
  // during the adaptive trial before all doses have been assigned)
  int n_obs_doses = dose_level.size();
   
  for (int l = 0; l < n_obs_doses; l++) {
    for (int k = 0; k < K[0]; k++) {
       
      // find indices where y_T == k and std_d == dose_level[l]
      std::vector<int> cur_i;
      for (int i = 0; i < n; i++) {
        if (y_T_vec[i] == k && std::abs(std_d_vec[i] - dose_level[l]) < 1e-10) {
          cur_i.push_back(i);
        }
      }
      
      if (cur_i.empty()) continue;
       
      NumericVector tb = t_bound[k];
       
      for (int idx : cur_i) {
         
        double cur_mu = mu_T[idx] + rho * (z_R[idx] - mu_R[idx]);
         
        double lower = R::pnorm(tb[0], cur_mu, cur_sd, 1, 0);
        double upper = R::pnorm(tb[1], cur_mu, cur_sd, 1, 0);
         
        double U = R::runif(lower, upper);
         
        new_z_T[idx] = R::qnorm(U, cur_mu, cur_sd, 1, 0);
      } 
    }
  }
  
  // z_R update
  NumericVector new_z_R(n, NA_REAL);
   
  for (int l = 0; l < n_obs_doses; l++) {
    for (int k = 0; k < K[1]; k++) {
       
      // find indices where y_R == k and std_d == dose_level[l]
      std::vector<int> cur_i;
      for (int i = 0; i < n; i++) {
        if (y_R_vec[i] == k && std::abs(std_d_vec[i] - dose_level[l]) < 1e-10) {
          cur_i.push_back(i);
        } 
      }
      
      if (cur_i.empty()) continue;
       
      NumericVector rb = r_bound[k];
       
      for (int idx : cur_i) {
         
        double cur_mu = mu_R[idx] + rho * (new_z_T[idx] - mu_T[idx]);
         
        double lower = R::pnorm(rb[0], cur_mu, cur_sd, 1, 0);
        double upper = R::pnorm(rb[1], cur_mu, cur_sd, 1, 0);
         
        double U = R::runif(lower, upper);
         
        new_z_R[idx] = R::qnorm(U, cur_mu, cur_sd, 1, 0);
     }
    }
  }
  
  // combine z_T and z_R
  NumericVector final_z(2 * n);
  for (int i = 0; i < n; i++) {
    final_z[i]     = new_z_T[i];
    final_z[n + i] = new_z_R[i];
  }
   
  return final_z;
} 

/////////////////////////////// b_0 update ///////////////////////////////////

// internal only
double samp_b_0_internal(
    NumericVector z,
    double b_1, double rho, double sig_2,
    double b_0_bar, double nu_0_2,
    NumericVector mu_R, NumericVector d,
    int nt)
{
  NumericVector z_T = z[Rcpp::seq(0, nt - 1)];
  NumericVector z_R = z[Rcpp::seq(nt, 2 * nt - 1)];
   
  double inv_var = 1.0 / (sig_2 * (1.0 - rho * rho));
   
  // b_star = sum(inv_var * (z_T - rho * (z_R - mu_R) - b_1 * d)) + b_0_bar / nu_0_2
  double b_star = 0.0;
  for (int i = 0; i < nt; i++) {
    b_star += inv_var * (z_T[i] - rho * (z_R[i] - mu_R[i]) - b_1 * d[i]);
  } 
  b_star += b_0_bar / nu_0_2;
   
  double sd_star = 1.0 / (nt * inv_var + 1.0 / nu_0_2);
   
  double mu = b_star * sd_star;
  
  return R::rnorm(mu, std::sqrt(sd_star));
} 

/////////////////////////////// b_1 update ////////////////////////////////////

// internal only
double samp_b_1_internal(
    NumericVector z,
    double b_0, double rho, double sig_2,
    double b_1_bar, double nu_1_2,
    NumericVector mu_R, NumericVector d,
    int nt)
{ 
  NumericVector z_T = z[Rcpp::seq(0, nt - 1)];
  NumericVector z_R = z[Rcpp::seq(nt, 2 * nt - 1)];
   
  double inv_var = 1.0 / (sig_2 * (1.0 - rho * rho));
   
  // b_star = sum(d / (sig_2 * (1 - rho^2)) * (z_T - rho * (z_R - mu_R) - b_0)) + b_1_bar / nu_1_2
  double b_star = 0.0;
  double sd_star_inv = 0.0;
  for (int i = 0; i < nt; i++) {
    b_star     += d[i] * inv_var * (z_T[i] - rho * (z_R[i] - mu_R[i]) - b_0);
    sd_star_inv += d[i] * d[i] * inv_var;
  } 
  b_star += b_1_bar / nu_1_2;
  sd_star_inv += 1.0 / nu_1_2;
   
  double sd_star = 1.0 / sd_star_inv;
  double mu = b_star * sd_star;
   
  // truncated normal - lower bound at 0
  double lower = R::pnorm(0.0, mu, std::sqrt(sd_star), 1, 0);
  double U     = R::runif(lower, 1.0);
   
  return R::qnorm(U, mu, std::sqrt(sd_star), 1, 0);
} 

////////////////////////////// lambda update //////////////////////////////////

// internal only
NumericVector samp_lamb_internal(
    NumericVector z, NumericVector y,
    double eta, NumericVector cur_lamb,
    int K)
{
  // no free cutpoints to sample when K <= 2 (return empty vector immediately)
  if (K <= 2) return NumericVector(0);

  NumericVector new_lamb(K - 2, NA_REAL);
  NumericVector z_max(K - 2, NA_REAL);
  NumericVector z_min(K - 1, NA_REAL);
   
  // compute z_max[k] = max(z[y == k]) for k = 1:(K-2)
  for (int k = 1; k <= K - 2; k++) {
    double cur_max = R_NegInf;
    for (int i = 0; i < z.size(); i++) {
      if (y[i] == k && z[i] > cur_max) cur_max = z[i];
    } 
    z_max[k - 1] = cur_max;
  }
   
  // compute z_min[k] = min(z[y == k]) for k = 1:(K-1)
  for (int k = 1; k <= K - 1; k++) {
    double cur_min = R_PosInf;
    for (int i = 0; i < z.size(); i++) {
      if (y[i] == k && z[i] < cur_min) cur_min = z[i];
    } 
    z_min[k - 1] = cur_min;
  }
   
  if (K == 3) {
    
    // single lambda: cutpoints (0, lamb_1)
    // lamb_1 > max(z where y == 1) and lamb_1 < min(z where y == 2)
    double lower = R::pgamma(z_max[0], 1.0, eta, 1, 0);
    double upper = R::pgamma(z_min[1], 1.0, eta, 1, 0);
    double U = R::runif(lower, upper);
    new_lamb[0] = R::qgamma(U, 1.0, eta, 1, 0);
    
  } else if (K == 4) {
    
    // lambda_1
    double lower = R::pgamma(
      std::max(z_max[0], z_max[1] - cur_lamb[1]), 1.0, eta, 1, 0);
    double upper = R::pgamma(
      std::min(z_min[1], z_min[2] - cur_lamb[1]), 1.0, eta, 1, 0);
     
    double U = R::runif(lower, upper);
    new_lamb[0] = R::qgamma(U, 1.0, eta, 1, 0);
     
    // lambda_2
    double lower_2 = R::pgamma(z_max[1] - cur_lamb[0], 1.0, eta, 1, 0);
    double upper_2 = R::pgamma(z_min[2] - cur_lamb[0], 1.0, eta, 1, 0);
     
    U = R::runif(lower_2, upper_2);
    new_lamb[1] = R::qgamma(U, 1.0, eta, 1, 0);
    
  } else if (K > 4) {
    
      // lambda_1
      // lamb_cs = cumsum(cur_lamb[2:(K-2)]), 0-indexed: cur_lamb[1:(K-3)]
      NumericVector lamb_cs(K - 3);
      lamb_cs[0] = cur_lamb[1];
      for (int k = 1; k < K - 3; k++) {
        lamb_cs[k] = lamb_cs[k - 1] + cur_lamb[k + 1];
      }
      
      // max(c(z_max[1], z_max[2:(K-2)] - lamb_cs))
      // 0-indexed: z_max[0], z_max[1:(K-3)] - lamb_cs[0:(K-4)]
      double max_val = z_max[0];
      for (int k = 0; k < K - 3; k++) {
        max_val = std::max(max_val, z_max[k + 1] - lamb_cs[k]);
      }
      
      // min(c(z_min[2], z_min[3:(K-1)] - lamb_cs))
      // 0-indexed: z_min[1], z_min[2:(K-2)] - lamb_cs[0:(K-4)]
      double min_val = z_min[1];
      for (int k = 0; k < K - 3; k++) {
        min_val = std::min(min_val, z_min[k + 2] - lamb_cs[k]);
      }
      
      double lower = R::pgamma(max_val, 1.0, eta, 1, 0);
      double upper = R::pgamma(min_val, 1.0, eta, 1, 0);
      double U     = R::runif(lower, upper);
      new_lamb[0]  = R::qgamma(U, 1.0, eta, 1, 0);
      
      // lambda k = 2:(K-2), 0-indexed: 1:(K-3)
      // cs_vec is (K-3) rows x (K-2) cols
      NumericMatrix cs_vec(K - 3, K - 2);
      
      for (int k = 1; k <= K - 3; k++) {  // k is 0-indexed lambda position
        
        // cumsum(cur_lamb[-k]), 0-indexed: skip element k
        NumericVector lamb_excl(K - 3);
        int idx = 0;
        for (int j = 0; j < K - 2; j++) {
          if (j != k) lamb_excl[idx++] = cur_lamb[j];
        }
        // store cumsum in column k of cs_vec
        cs_vec(0, k) = lamb_excl[0];
        for (int j = 1; j < K - 3; j++) {
          cs_vec(j, k) = cs_vec(j - 1, k) + lamb_excl[j];
        }
        
        // max(z_max[k:(K-3)] - cs_vec[(k-1):(K-4), k])
        // 0-indexed: z_max[k:(K-3)], cs_vec rows (k-1):(K-4)
        double max_v = R_NegInf;
        for (int j = 0; j < K - 2 - k; j++) {
          max_v = std::max(max_v, z_max[k + j] - cs_vec(k - 1 + j, k));
        }
        
        // min(z_min[(k+1):(K-2)] - cs_vec[(k-1):(K-4), k])
        // 0-indexed: z_min[(k+1):(K-2)], cs_vec rows (k-1):(K-4)
        double min_v = R_PosInf;
        for (int j = 0; j < K - 2 - k; j++) {
          min_v = std::min(min_v, z_min[k + 1 + j] - cs_vec(k - 1 + j, k));
        }
        
        lower       = R::pgamma(max_v, 1.0, eta, 1, 0);
        upper       = R::pgamma(min_v, 1.0, eta, 1, 0);
        U           = R::runif(lower, upper);
        new_lamb[k] = R::qgamma(U, 1.0, eta, 1, 0);
      }
    }
   
  
  return new_lamb;
} 


/////////////////////// MH Target Functions ///////////////////////////////////

// internal only
double alpha_log_target_internal(
    NumericVector a,
    NumericVector z,
    double rho, double sig_2,
    NumericVector w_2,
    NumericVector a_bar,
    int a_index,
    NumericVector mu_T,
    NumericVector d,
    int nt)
{
  NumericVector z_T = z[Rcpp::seq(0, nt - 1)];
  NumericVector z_R = z[Rcpp::seq(nt, 2 * nt - 1)];
  
  // mu_R = a[1] + a[4] / (1 + exp(-a[2] * d + a[3]))
  // 0-indexed: a[0] + a[3] / (1 + exp(-a[1] * d + a[2]))
  NumericVector mu_R = a[0] + a[3] / (1.0 + Rcpp::exp(-a[1] * d + a[2]));
  
  double inv_var = 1.0 / (2.0 * sig_2 * (1.0 - rho * rho));
  
  // sum(-1/(2*sig_2*(1-rho^2)) * (mu_R^2 - 2*mu_R*(rho*(mu_T - z_T) + z_R)))
  double log_target = 0.0;
  for (int i = 0; i < nt; i++) {
    log_target += -inv_var * (mu_R[i] * mu_R[i] -
      2.0 * mu_R[i] * (rho * (mu_T[i] - z_T[i]) + z_R[i]));
  }
  
  // prior term - 0-indexed: a_index - 1
  log_target -= 1.0 / (2.0 * w_2[a_index - 1]) *
    std::pow(a[a_index - 1] - a_bar[a_index - 1], 2.0);
  
  return log_target;
}

// internal only
double a13_log_jacob_internal(double alpha)
{
  return -std::log(alpha);
}

// internal only
double rho_log_target_internal(
    double rho_til,
    NumericVector z,
    double sig_2,
    double a_rho, double b_rho,
    NumericVector mu_T, NumericVector mu_R,
    int nt)
{
  NumericVector z_T = z[Rcpp::seq(0, nt - 1)];
  NumericVector z_R = z[Rcpp::seq(nt, 2 * nt - 1)];
  
  double rho = rho_til * 2.0 - 1.0;
  
  double sum_norm = 0.0;
  for (int i = 0; i < nt; i++) {
    double dT = z_T[i] - mu_T[i];
    double dR = z_R[i] - mu_R[i];
    sum_norm += dT * dT + dR * dR - 2.0 * rho * dT * dR;
  }
  
  double log_target = - nt / 2.0 * std::log(1.0 - rho * rho) -
    1.0 / (2.0 * sig_2 * (1.0 - rho * rho)) * sum_norm +
    (a_rho - 1.0) * std::log(rho_til) +
    (b_rho - 1.0) * std::log(1.0 - rho_til);
  
  return log_target;
}

// internal only
double rho_log_jacob_internal(double rho)
{
  return std::log(rho * (1.0 - rho));
}

/////////////// Guarding against NAs from runaway sampler ////////////////////


inline int which_max_finite(const Rcpp::NumericVector& x) {
  int best = -1;
  for (int i = 0; i < x.size(); i++) {
    if (!R_finite(x[i])) continue;
    if (best < 0 || x[i] > x[best]) best = i;
  }
  return best;   // -1 if nothing finite
}



/////////////////////// Posterior Expected Utility ////////////////////////////

// internal only
NumericVector EU_internal(
    List pi_jk,
    NumericMatrix utility_mat,
    int n_doses)
{
  NumericVector expected_utility(n_doses);
  
  for (int l = 0; l < n_doses; l++) {
    NumericMatrix pi_l = pi_jk[l];
    double eu = 0.0;
    for (int i = 0; i < pi_l.nrow(); i++) {
      for (int j = 0; j < pi_l.ncol(); j++) {
        eu += pi_l(i, j) * utility_mat(i, j);
      }
    }
    expected_utility[l] = eu;
  }
  
  return expected_utility;
}

//[[Rcpp::export]]
NumericVector Exp_U(
    List pi_jk,
    NumericMatrix utility_mat,
    int n_doses)
{
  return EU_internal(pi_jk, utility_mat, n_doses);
}


// [[Rcpp::export]]
List run_sampler(
    int N, int burn, int thin_by,
    NumericVector y_T_vec, NumericVector y_R_vec,
    NumericVector d,
    NumericVector doses,
    int nt, int n_doses,
    double cur_a_0, double cur_a_1, double cur_a_2, double cur_a_3,
    double cur_b_0, double cur_b_1,
    NumericVector cur_lamb_T, NumericVector cur_lamb_R,
    double cur_rho,
    NumericVector cur_z,
    NumericVector mu_T, NumericVector mu_R,
    NumericVector alpha_bar, NumericVector w,
    NumericVector beta_bar, NumericVector nu,
    double eta_T, double eta_R,
    double a_rho, double b_rho,
    double sig_2,
    IntegerVector K,
    NumericMatrix Ut_mat,
    NumericVector doses_mg)
{
  double sig_sd = std::sqrt(sig_2);
   
  // initialise storage
  int n_params  = 4 + 2 + cur_lamb_T.size() + cur_lamb_R.size() + 1;
  int n_samples = (int)std::floor((double)(N - burn) / thin_by);
   
  NumericMatrix theta_coef(N, n_params);
  NumericMatrix fin_theta(n_samples, n_params);
  NumericMatrix exp_ut(n_samples, n_doses);
  NumericVector optimal_dose(n_samples);
  List post_pi_jk(n_samples);
   
  int i_sam = 0;
   
   for (int t = 1; t < N; t++) {
     
    // update z
    cur_z = samp_z_internal(
      cur_rho, cur_lamb_T, cur_lamb_R,
      cur_z[Rcpp::seq(nt, 2 * nt - 1)],
           y_T_vec, y_R_vec, d,
           mu_T, mu_R, sig_2, n_doses, K);
     
    // update b_0
    cur_b_0 = samp_b_0_internal(
      cur_z, cur_b_1, cur_rho, sig_2,
      beta_bar[0], nu[0],
                     mu_R, d, nt);
     
    // update b_1
    cur_b_1 = samp_b_1_internal(
      cur_z, cur_b_0, cur_rho, sig_2,
      beta_bar[1], nu[1],
                     mu_R, d, nt);
     
    // update mean toxicity
    mu_T = cur_b_0 + cur_b_1 * d;
     
    // update lambda_T
    cur_lamb_T = samp_lamb_internal(
      cur_z[Rcpp::seq(0, nt - 1)],
           y_T_vec, eta_T, cur_lamb_T, K[0]);
     
    // update lambda_R
    cur_lamb_R = samp_lamb_internal(
      cur_z[Rcpp::seq(nt, 2 * nt - 1)],
           y_R_vec, eta_R, cur_lamb_R, K[1]);
     
    // update alpha_0 - normal proposal
    double a_0_prop = R::rnorm(cur_a_0, 1.5);
     
    double log_A_num = alpha_log_target_internal(
      NumericVector::create(a_0_prop, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 1, mu_T, d, nt);
     
    double log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 1, mu_T, d, nt);
     
    double log_ratio = std::min(0.0, log_A_num - log_A_denom);
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_0 = a_0_prop;
     
    // update alpha_1 - log-normal proposal
    double a_1_star = R::rnorm(std::log(cur_a_1), 0.3);
    double a_1_prop = std::exp(a_1_star);
     
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, a_1_prop, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 2, mu_T, d, nt);
     
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 2, mu_T, d, nt);
     
    log_ratio = std::min(0.0, log_A_num + a13_log_jacob_internal(cur_a_1)
                            - log_A_denom - a13_log_jacob_internal(a_1_prop));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_1 = a_1_prop;
   
    // update alpha_2 - normal proposal
    double a_2_prop = R::rnorm(cur_a_2, 0.9);
     
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, a_2_prop, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 3, mu_T, d, nt);
     
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 3, mu_T, d, nt);
   
    log_ratio = std::min(0.0, log_A_num - log_A_denom);
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_2 = a_2_prop;
    
    // update alpha_3 - log-normal proposal
    double a_3_star = R::rnorm(std::log(cur_a_3), 0.4);
    double a_3_prop = std::exp(a_3_star);
   
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, a_3_prop),
      cur_z, cur_rho, sig_2, w, alpha_bar, 4, mu_T, d, nt);
    
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 4, mu_T, d, nt);
    
    log_ratio = std::min(0.0, log_A_num + a13_log_jacob_internal(cur_a_3)
                           - log_A_denom - a13_log_jacob_internal(a_3_prop));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_3 = a_3_prop;
   
    // update mean response
    mu_R = cur_a_0 + cur_a_3 / (1.0 + Rcpp::exp(-cur_a_1 * d + cur_a_2));
    
    // update rho - logit-transformed proposal
    double rho_til  = (cur_rho + 1.0) / 2.0;
    double rho_star = std::log(rho_til / (1.0 - rho_til));
    double xi_star  = R::rnorm(rho_star, 0.85);
    double rho_prop = std::exp(xi_star) / (1.0 + std::exp(xi_star));
   
    log_A_num = rho_log_target_internal(
      rho_prop, cur_z, sig_2, a_rho, b_rho, mu_T, mu_R, nt);
    
    log_A_denom = rho_log_target_internal(
      rho_til, cur_z, sig_2, a_rho, b_rho, mu_T, mu_R, nt);
   
    log_ratio = std::min(0.0, log_A_num + rho_log_jacob_internal(rho_prop)
                          - log_A_denom - rho_log_jacob_internal(rho_til));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_rho = 2.0 * rho_prop - 1.0;
    
    // save draws to theta_coef
    int col = 0;
    theta_coef(t, col++) = cur_a_0;
    theta_coef(t, col++) = cur_a_1;
    theta_coef(t, col++) = cur_a_2;
    theta_coef(t, col++) = cur_a_3;
    theta_coef(t, col++) = cur_b_0;
    theta_coef(t, col++) = cur_b_1;
    for (int k = 0; k < cur_lamb_T.size(); k++) theta_coef(t, col++) = cur_lamb_T[k];
    for (int k = 0; k < cur_lamb_R.size(); k++) theta_coef(t, col++) = cur_lamb_R[k];
    theta_coef(t, col++) = cur_rho;
    
    // store posterior samples after burn-in
    if ((t + 1) > burn && ((t + 1) % thin_by) == 0) {
      
      col = 0;
      fin_theta(i_sam, col++) = cur_a_0;
      fin_theta(i_sam, col++) = cur_a_1;
      fin_theta(i_sam, col++) = cur_a_2;
      fin_theta(i_sam, col++) = cur_a_3;
      fin_theta(i_sam, col++) = cur_b_0;
      fin_theta(i_sam, col++) = cur_b_1;
      for (int k = 0; k < cur_lamb_T.size(); k++) fin_theta(i_sam, col++) = cur_lamb_T[k];
      for (int k = 0; k < cur_lamb_R.size(); k++) fin_theta(i_sam, col++) = cur_lamb_R[k];
      fin_theta(i_sam, col++) = cur_rho;
      
      // posterior joint probabilities
      List cur_post_pi = post_pi_jk_fxn_internal(
        cur_lamb_T, cur_lamb_R,
        cur_a_0, cur_a_1, cur_a_2, cur_a_3,
        cur_b_0, cur_b_1,
        sig_sd, cur_rho,
        doses, n_doses, K);
     
      post_pi_jk[i_sam] = cur_post_pi;
       
      // expected utility
      NumericVector eu = EU_internal(cur_post_pi, Ut_mat, n_doses);
      for (int j = 0; j < n_doses; j++) exp_ut(i_sam, j) = eu[j];
       
      // optimal dose index
      int im = which_max_finite(eu);
      optimal_dose[i_sam] = (im < 0) ? NA_REAL : doses_mg[im];
      
      i_sam++;
    }
    
  }
   
  return List::create(
    Rcpp::Named("theta_coef")   = theta_coef,
    Rcpp::Named("fin_theta")    = fin_theta,
    Rcpp::Named("exp_ut")       = exp_ut,
    Rcpp::Named("optimal_dose") = optimal_dose,
    Rcpp::Named("post_pi_jk")   = post_pi_jk
  );
} 

// internal only
double U_low_fxn_internal(
    NumericVector h_T, NumericVector h_R,
    double sigma,
    NumericMatrix Ut_mat,
    IntegerVector K)
{
  int B = 5000;
   
  // compute means from upper tail quantiles
  double h_T1 = 1.0 - h_T[0];
  double h_R1 = 1.0 - h_R[0];
   
  double mu_T = -R::qnorm(h_T1, 0.0, 1.0, 1, 0) * sigma;
  double mu_R = -R::qnorm(h_R1, 0.0, 1.0, 1, 0) * sigma;
   
  // compute cutpoints: e_T = c(-Inf, qnorm(h_T, mu_T, sigma, lower.tail = FALSE), Inf)
  NumericVector e_T(h_T.size() + 2);
  e_T[0] = R_NegInf;
  for (int k = 0; k < h_T.size(); k++) {
    e_T[k + 1] = R::qnorm(h_T[k], mu_T, sigma, 0, 0);  // lower.tail = FALSE
  } 
  e_T[h_T.size() + 1] = R_PosInf;
   
  NumericVector e_R(h_R.size() + 2);
  e_R[0] = R_NegInf;
  for (int k = 0; k < h_R.size(); k++) {
    e_R[k + 1] = R::qnorm(h_R[k], mu_R, sigma, 0, 0);
  } 
  e_R[h_R.size() + 1] = R_PosInf;
   
  // bounds
  List t_bound(K[0]);
  List r_bound(K[1]);
    
  for (int k = 0; k < K[0]; k++) {
    t_bound[k] = NumericVector::create(e_T[k], e_T[k + 1]);
  }  
  for (int k = 0; k < K[1]; k++) {
    r_bound[k] = NumericVector::create(e_R[k], e_R[k + 1]);
  }
   
  // rho_B = seq(from = 0, to = 0.9, length.out = B)
  NumericVector rho_B(B);
  double step = 0.9 / (B - 1);
  for (int b = 0; b < B; b++) {
    rho_B[b] = b * step;
  }
   
  // accumulate expected utility over B values of rho
  double sum_eu = 0.0;
   
  for (int b = 0; b < B; b++) {
     
    // joint_prob_internal expects sig1 (sd) and rho
    // sigma is already the standard deviation in this function
    NumericMatrix pi_jk = joint_prob_internal(
      t_bound, r_bound, mu_T, mu_R, sigma, rho_B[b], K);
     
    // sum(pi_jk * Ut_mat)
    double eu = 0.0;
    for (int i = 0; i < K[0]; i++) {
      for (int j = 0; j < K[1]; j++) {
        eu += pi_jk(i, j) * Ut_mat(i, j);
      }
    } 
    sum_eu += eu;
  }
    
  return sum_eu / B;
} 

// [[Rcpp::export]]
double U_low_fxn(
    NumericVector h_T, NumericVector h_R,
    double sigma,
    NumericMatrix Ut_mat,
    IntegerVector K)
{ 
  return U_low_fxn_internal(h_T, h_R, sigma, Ut_mat, K);
} 



/////////////////////// Sampling Observations /////////////////////////////////

// The three routines below reproduce R's sample(..., prob = ) bit for bit.
// R does not use a plain inverse-CDF walk: it first sorts the probabilities
// into descending order with revsort() (an unstable heapsort that permutes an
// index array alongside), then walks the sorted cumulative sum. Drawing from
// the unsorted vector consumes the same number of uniforms but can select a
// different element, so the sorted version is required to match R exactly.
// These are called once per cohort, never inside the MCMC loop, so the extra
// O(n log n) sort on a length-(K[0]*K[1]) vector is negligible.

// Verbatim port of revsort() from R's src/main/sort.c.
// Sorts a[] into descending order and permutes ib[] alongside.
static void revsort_internal(double *a, int *ib, int n)
{
  int l, j, ir, i;
  double ra;
  int ii;

  if (n <= 1) return;

  a--; ib--;

  l = (n >> 1) + 1;
  ir = n;

  for (;;) {
    if (l > 1) {
      l = l - 1;
      ra = a[l];
      ii = ib[l];
    } else {
      ra = a[ir];
      ii = ib[ir];
      a[ir] = a[1];
      ib[ir] = ib[1];
      if (--ir == 1) {
        a[1] = ra;
        ib[1] = ii;
        return;
      }
    }
    i = l;
    j = l << 1;
    while (j <= ir) {
      if (j < ir && a[j] > a[j + 1]) ++j;
      if (ra > a[j]) {
        a[i] = a[j];
        ib[i] = ib[j];
        j += (i = j);
      } else {
        j = ir + 1;
      }
      a[i] = ra;
      ib[i] = ii;
    }
  }
}

// Port of ProbSampleReplace() from R's src/main/random.c, preceded by the
// normalisation that FixupProb() performs. Returns nans draws as 1-indexed
// positions in prob_in. Equivalent to
//   sample(seq_along(prob_in), nans, replace = TRUE, prob = prob_in)
// Note: R's sampler calls unif_rand() directly, not runif(0, 1). Rf_runif
// rejects draws of exactly 0 or 1 in a do-while loop, which would desynchronise
// the stream, so unif_rand() must be used here.
IntegerVector prob_sample_replace_internal(NumericVector prob_in, int nans)
{
  int n = prob_in.size();
  std::vector<double> p(n);
  std::vector<int> perm(n);

  double sum = 0.0;
  for (int i = 0; i < n; i++) { p[i] = prob_in[i]; sum += p[i]; }
  for (int i = 0; i < n; i++) p[i] /= sum;
  for (int i = 0; i < n; i++) perm[i] = i + 1;

  revsort_internal(&p[0], &perm[0], n);

  for (int i = 1; i < n; i++) p[i] += p[i - 1];

  IntegerVector ans(nans);
  int nm1 = n - 1;
  for (int i = 0; i < nans; i++) {
    double rU = ::unif_rand();
    int j;
    for (j = 0; j < nm1; j++) if (rU <= p[j]) break;
    ans[i] = perm[j];
  }
  return ans;
}

// Port of ProbSampleNoReplace() from R's src/main/random.c specialised to a
// single draw. Returns a 1-indexed position in prob_in. Equivalent to
//   sample(seq_along(prob_in), 1, prob = prob_in)
int prob_sample_no_replace_one_internal(NumericVector prob_in)
{
  int n = prob_in.size();
  std::vector<double> p(n);
  std::vector<int> perm(n);

  double sum = 0.0;
  for (int i = 0; i < n; i++) { p[i] = prob_in[i]; sum += p[i]; }
  for (int i = 0; i < n; i++) p[i] /= sum;
  for (int i = 0; i < n; i++) perm[i] = i + 1;

  revsort_internal(&p[0], &perm[0], n);

  double rT = ::unif_rand();
  double mass = 0.0;
  int n1 = n - 1;
  int j;
  for (j = 0; j < n1; j++) {
    mass += p[j];
    if (rT <= mass) break;
  }
  return perm[j];
}

// Internal only - sample one element of vec with weights prob.
// Mirrors R's sample(vec, 1, prob = prob/sum(prob)).
int sample_with_prob_internal(IntegerVector vec, NumericVector prob)
{
  int n = vec.size();
  if (n == 1) return vec[0];
  return vec[prob_sample_no_replace_one_internal(prob) - 1];
}

// Internal only - simulate cohort_sz observations at the (1-indexed) assigned
// dose and append them to y_cur. Mirrors the shared observation-sampling block
// of the R assignment functions.
// y matrix has 4 columns: y_T, y_R, Dose, std_d
NumericMatrix sim_cohort_internal(
    int cohort_sz,
    int asn_dose_idx,
    NumericVector doses,
    NumericVector std_d_doses,
    List pi_jk,
    NumericMatrix y_cur,
    IntegerVector K)
{
  int n_cells = K[0] * K[1];
  NumericMatrix pi_l = pi_jk[asn_dose_idx - 1];

  // flatten column-major so the cell index matches R's expand.grid ordering
  // (y_T varies fastest, then y_R) and R's coercion of the matrix passed as prob
  NumericVector prob(n_cells);
  for (int r = 0; r < K[1]; r++) {
    for (int t = 0; t < K[0]; t++) {
      prob[r * K[0] + t] = pi_l(t, r);
    }
  }

  // all cohort_sz cells in one call, matching
  // sample(1:(K[1]*K[2]), cohort_sz, replace = TRUE, prob = pi_jk[[asn_dose]])
  IntegerVector cells = prob_sample_replace_internal(prob, cohort_sz);

  NumericMatrix y_new(cohort_sz, 4);

  for (int s = 0; s < cohort_sz; s++) {

    int cell = cells[s] - 1;

    y_new(s, 0) = cell % K[0];                   // y_T
    y_new(s, 1) = cell / K[0];                   // y_R
    y_new(s, 2) = doses[asn_dose_idx - 1];       // Dose
    y_new(s, 3) = std_d_doses[asn_dose_idx - 1]; // std_d
  }

  // append to y_cur
  int n_cur = y_cur.nrow();
  NumericMatrix y_out(n_cur + cohort_sz, 4);
  for (int i = 0; i < n_cur; i++) {
    for (int j = 0; j < 4; j++) y_out(i, j) = y_cur(i, j);
  }
  for (int i = 0; i < cohort_sz; i++) {
    for (int j = 0; j < 4; j++) y_out(n_cur + i, j) = y_new(i, j);
  }

  Rcpp::colnames(y_out) = Rcpp::CharacterVector::create(
    "y_T", "y_R", "Dose", "std_d");

  return y_out;
}

// Internal only - U-Bayes urn dose assignment.
// A_t: 1-indexed acceptable dose indices (in increasing dose order)
// urn: modified in place
// Returns the 1-indexed assigned dose, or -1 if A_t is empty.
int ub_sample_assign_internal(
    IntegerVector A_t,
    IntegerVector& urn,
    int n_doses,
    int n_balls,
    int coh_num)
{
  int n_A = A_t.size();
  if (n_A == 0) return -1;

  int assign;

  if (coh_num == 0) {
    // first cohort always starts at the second dose
    assign = 2;
  } else if (n_A > 1) {
    NumericVector red_urn(n_A);
    for (int i = 0; i < n_A; i++) red_urn[i] = urn[A_t[i] - 1];
    assign = sample_with_prob_internal(A_t, red_urn);
  } else {
    assign = A_t[0];
  }

  // update urn: urn[-d_ind] <- urn[-d_ind] + n_balls
  for (int l = 1; l <= n_doses; l++) {
    if (l != assign) urn[l - 1] += n_balls;
  }

  return assign;
}

/////////////////////// internal sampler //////////////////////////////////////

// Internal only - runs the Gibbs/MH sampler on the current y matrix and returns
// the same list structure as the exported run_sampler.
List run_sampler_internal(
    int N, int burn, int thin_by,
    NumericMatrix y,
    int n_doses,
    double cur_a_0, double cur_a_1, double cur_a_2, double cur_a_3,
    double cur_b_0, double cur_b_1,
    NumericVector cur_lamb_T, NumericVector cur_lamb_R,
    double cur_rho,
    NumericVector alpha_bar, NumericVector w,
    NumericVector beta_bar, NumericVector nu,
    double eta_T, double eta_R,
    double a_rho, double b_rho,
    double sig_2,
    IntegerVector K,
    NumericMatrix Ut_mat,
    NumericVector doses_mg,
    NumericVector std_d_doses)
{
  int n_t = y.nrow();

  NumericVector y_T_vec(n_t), y_R_vec(n_t), d(n_t);
  for (int i = 0; i < n_t; i++) {
    y_T_vec[i] = y(i, 0);
    y_R_vec[i] = y(i, 1);
    d[i]       = y(i, 3);
  }

  double sig_sd = std::sqrt(sig_2);

  // initial z
  NumericVector cur_z(2 * n_t);
  for (int i = 0; i < 2 * n_t; i++) cur_z[i] = R::rnorm(0.0, 1.0);

  // initial mu
  NumericVector mu_T(n_t), mu_R(n_t);
  for (int i = 0; i < n_t; i++) {
    mu_T[i] = cur_b_0 + cur_b_1 * d[i];
    mu_R[i] = cur_a_0 + cur_a_3 / (1.0 + std::exp(-cur_a_1 * d[i] + cur_a_2));
  }

  int n_params  = 4 + 2 + cur_lamb_T.size() + cur_lamb_R.size() + 1;
  int n_samples = (int)std::floor((double)(N - burn) / thin_by);

  NumericMatrix theta_coef(N, n_params);
  NumericMatrix fin_theta(n_samples, n_params);
  NumericMatrix exp_ut(n_samples, n_doses);
  NumericVector optimal_dose(n_samples);
  List post_pi_jk(n_samples);

  int i_sam = 0;

  for (int t = 1; t < N; t++) {

    cur_z = samp_z_internal(
      cur_rho, cur_lamb_T, cur_lamb_R,
      cur_z[Rcpp::seq(n_t, 2 * n_t - 1)],
      y_T_vec, y_R_vec, d,
      mu_T, mu_R, sig_2, n_doses, K);

    cur_b_0 = samp_b_0_internal(
      cur_z, cur_b_1, cur_rho, sig_2,
      beta_bar[0], nu[0], mu_R, d, n_t);

    cur_b_1 = samp_b_1_internal(
      cur_z, cur_b_0, cur_rho, sig_2,
      beta_bar[1], nu[1], mu_R, d, n_t);

    mu_T = cur_b_0 + cur_b_1 * d;

    cur_lamb_T = samp_lamb_internal(
      cur_z[Rcpp::seq(0, n_t - 1)],
      y_T_vec, eta_T, cur_lamb_T, K[0]);

    cur_lamb_R = samp_lamb_internal(
      cur_z[Rcpp::seq(n_t, 2 * n_t - 1)],
      y_R_vec, eta_R, cur_lamb_R, K[1]);

    // alpha_0
    double a_0_prop = R::rnorm(cur_a_0, 1.5);
    double log_A_num = alpha_log_target_internal(
      NumericVector::create(a_0_prop, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 1, mu_T, d, n_t);
    double log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 1, mu_T, d, n_t);
    double log_ratio = std::min(0.0, log_A_num - log_A_denom);
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_0 = a_0_prop;

    // alpha_1
    double a_1_star = R::rnorm(std::log(cur_a_1), 0.3);
    double a_1_prop = std::exp(a_1_star);
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, a_1_prop, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 2, mu_T, d, n_t);
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 2, mu_T, d, n_t);
    log_ratio = std::min(0.0, log_A_num + a13_log_jacob_internal(cur_a_1)
                              - log_A_denom - a13_log_jacob_internal(a_1_prop));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_1 = a_1_prop;

    // alpha_2
    double a_2_prop = R::rnorm(cur_a_2, 0.9);
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, a_2_prop, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 3, mu_T, d, n_t);
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 3, mu_T, d, n_t);
    log_ratio = std::min(0.0, log_A_num - log_A_denom);
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_2 = a_2_prop;

    // alpha_3
    double a_3_star = R::rnorm(std::log(cur_a_3), 0.4);
    double a_3_prop = std::exp(a_3_star);
    log_A_num = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, a_3_prop),
      cur_z, cur_rho, sig_2, w, alpha_bar, 4, mu_T, d, n_t);
    log_A_denom = alpha_log_target_internal(
      NumericVector::create(cur_a_0, cur_a_1, cur_a_2, cur_a_3),
      cur_z, cur_rho, sig_2, w, alpha_bar, 4, mu_T, d, n_t);
    log_ratio = std::min(0.0, log_A_num + a13_log_jacob_internal(cur_a_3)
                              - log_A_denom - a13_log_jacob_internal(a_3_prop));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_a_3 = a_3_prop;

    mu_R = cur_a_0 + cur_a_3 / (1.0 + Rcpp::exp(-cur_a_1 * d + cur_a_2));

    // rho
    double rho_til  = (cur_rho + 1.0) / 2.0;
    double rho_star = std::log(rho_til / (1.0 - rho_til));
    double xi_star  = R::rnorm(rho_star, 0.85);
    double rho_prop = std::exp(xi_star) / (1.0 + std::exp(xi_star));

    log_A_num = rho_log_target_internal(
      rho_prop, cur_z, sig_2, a_rho, b_rho, mu_T, mu_R, n_t);
    log_A_denom = rho_log_target_internal(
      rho_til, cur_z, sig_2, a_rho, b_rho, mu_T, mu_R, n_t);
    log_ratio = std::min(0.0, log_A_num + rho_log_jacob_internal(rho_prop)
                              - log_A_denom - rho_log_jacob_internal(rho_til));
    if (log_ratio >= std::log(R::runif(0.0, 1.0))) cur_rho = 2.0 * rho_prop - 1.0;

    // store every iteration
    int col = 0;
    theta_coef(t, col++) = cur_a_0;
    theta_coef(t, col++) = cur_a_1;
    theta_coef(t, col++) = cur_a_2;
    theta_coef(t, col++) = cur_a_3;
    theta_coef(t, col++) = cur_b_0;
    theta_coef(t, col++) = cur_b_1;
    for (int k = 0; k < cur_lamb_T.size(); k++) theta_coef(t, col++) = cur_lamb_T[k];
    for (int k = 0; k < cur_lamb_R.size(); k++) theta_coef(t, col++) = cur_lamb_R[k];
    theta_coef(t, col++) = cur_rho;

    // store thinned post-burn-in samples
    if ((t + 1) > burn && ((t + 1) % thin_by) == 0 && i_sam < n_samples) {

      col = 0;
      fin_theta(i_sam, col++) = cur_a_0;
      fin_theta(i_sam, col++) = cur_a_1;
      fin_theta(i_sam, col++) = cur_a_2;
      fin_theta(i_sam, col++) = cur_a_3;
      fin_theta(i_sam, col++) = cur_b_0;
      fin_theta(i_sam, col++) = cur_b_1;
      for (int k = 0; k < cur_lamb_T.size(); k++) fin_theta(i_sam, col++) = cur_lamb_T[k];
      for (int k = 0; k < cur_lamb_R.size(); k++) fin_theta(i_sam, col++) = cur_lamb_R[k];
      fin_theta(i_sam, col++) = cur_rho;

      List cur_post_pi = post_pi_jk_fxn_internal(
        cur_lamb_T, cur_lamb_R,
        cur_a_0, cur_a_1, cur_a_2, cur_a_3,
        cur_b_0, cur_b_1,
        sig_sd, cur_rho,
        std_d_doses, n_doses, K);

      post_pi_jk[i_sam] = cur_post_pi;

      NumericVector eu = EU_internal(cur_post_pi, Ut_mat, n_doses);
      for (int j = 0; j < n_doses; j++) exp_ut(i_sam, j) = eu[j];
      
      int im = which_max_finite(eu);
      optimal_dose[i_sam] = (im < 0) ? NA_REAL : doses_mg[im];

      i_sam++;
    }
    
  }

  return List::create(
    Rcpp::Named("theta_coef")   = theta_coef,
    Rcpp::Named("fin_theta")    = fin_theta,
    Rcpp::Named("exp_ut")       = exp_ut,
    Rcpp::Named("optimal_dose") = optimal_dose,
    Rcpp::Named("post_pi_jk")   = post_pi_jk);
}

/////////////////////// trial: U-Bayes urn assignment /////////////////////////

// One replicate of the U-Bayes adaptive trial (the body of the R for-loop over
// s in 1:num_trial). The caller supplies the seed and loops over replicates.
//
// Arguments
//   co_sz       - cohort size
//   n_doses     - number of dose levels
//   doses       - dose values in mg, increasing
//   std_d_doses - standardised doses, same order as doses
//   pi_jk       - list of n_doses true joint probability matrices (K[0] x K[1])
//   N_max       - maximum total sample size
//   cu_star     - final monitoring cut-off
//   U_low       - utility lower bound
//   n_balls     - urn increment for unassigned doses

// [[Rcpp::export]]
List run_trial_ubr(
    int co_sz,
    int n_doses,
    NumericVector doses,
    NumericVector std_d_doses,
    List pi_jk,
    int N_max,
    double cu_star,
    double U_low,
    int n_balls,
    int N, int burn, int thin_by,
    NumericVector alpha_bar, NumericVector w,
    NumericVector beta_bar, NumericVector nu,
    Rcpp::Nullable<double> eta_T, Rcpp::Nullable<double> eta_R,
    double a_rho, double b_rho,
    double sig_2,
    IntegerVector K,
    NumericMatrix Ut_mat,
    Rcpp::Nullable<Rcpp::List> init_values = R_NilValue)
{
  // eta placeholders are unused when the corresponding K = 2
  double eta_T_val = eta_T.isNotNull() ? Rcpp::as<double>(eta_T) : 1.0;
  double eta_R_val = eta_R.isNotNull() ? Rcpp::as<double>(eta_R) : 1.0;

  // initial values: defaults, overridden by any supplied in init_values
  double init_a_0 = 0.0, init_a_1 = 1.0, init_a_2 = 0.0, init_a_3 = 1.0;
  double init_b_0 = 0.0, init_b_1 = 1.0;
  double init_rho = 0.0;
  NumericVector init_lamb_T(std::max(0, K[0] - 2), 1.0);
  NumericVector init_lamb_R(std::max(0, K[1] - 2), 1.0);

  if (init_values.isNotNull()) {
    Rcpp::List iv(init_values);
    if (iv.containsElementNamed("cur_a_0")) init_a_0 = Rcpp::as<double>(iv["cur_a_0"]);
    if (iv.containsElementNamed("cur_a_1")) init_a_1 = Rcpp::as<double>(iv["cur_a_1"]);
    if (iv.containsElementNamed("cur_a_2")) init_a_2 = Rcpp::as<double>(iv["cur_a_2"]);
    if (iv.containsElementNamed("cur_a_3")) init_a_3 = Rcpp::as<double>(iv["cur_a_3"]);
    if (iv.containsElementNamed("cur_b_0")) init_b_0 = Rcpp::as<double>(iv["cur_b_0"]);
    if (iv.containsElementNamed("cur_b_1")) init_b_1 = Rcpp::as<double>(iv["cur_b_1"]);
    if (iv.containsElementNamed("cur_rho")) init_rho = Rcpp::as<double>(iv["cur_rho"]);
    if (iv.containsElementNamed("cur_lamb_T")) {
      SEXP v = iv["cur_lamb_T"];
      init_lamb_T = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
    if (iv.containsElementNamed("cur_lamb_R")) {
      SEXP v = iv["cur_lamb_R"];
      init_lamb_R = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
  }

  // initialise trial state
  int n_t = 0;
  int coh_num = 0;

  IntegerVector dose_count(n_doses);
  
  IntegerVector urn(n_doses);
  for (int l = 0; l < n_doses; l++) urn[l] = 1;

  NumericMatrix y(0, 4);
  Rcpp::colnames(y) = Rcpp::CharacterVector::create(
    "y_T", "y_R", "Dose", "std_d");

  // acceptable-dose indicators, one row per planned cohort
  int max_cohorts = (int)std::ceil((double)N_max / co_sz);
  Rcpp::LogicalMatrix fin_A_t(max_cohorts, n_doses);
  std::fill(fin_A_t.begin(), fin_A_t.end(), NA_LOGICAL);

  // A_t as 1-indexed dose positions; starts as all doses
  IntegerVector A_t(n_doses);
  for (int l = 0; l < n_doses; l++) A_t[l] = l + 1;

  IntegerVector unac_dose_i(0);
  List samp;
  bool sampler_ran = false;
  
  // "max_reached" or "all_unacceptable". Unlike the no-skip designs there is no
  // "no_assignable" state: assignment draws from all of A_t with no restriction
  // to doses at or below d_M(t), so a non-empty A_t is always assignable.
  Rcpp::String stop_reason = "max_reached";
  
  while (true) {
    
    // assign a dose, simulate the cohort, update the urn
    int asn = no_skip_v2_assign_internal(A_t, urn, y, doses, n_doses, n_balls, 
                                         coh_num);
    
    if (asn == -2) { stop_reason = "no_assignable"; break; }
    if (asn == -3) { Rcpp::stop("Internal error: empty y with coh_num > 0."); }
    if (asn < 0)   { stop_reason = "all_unacceptable"; break; }
    
    dose_count[asn - 1]++;

    y = sim_cohort_internal(co_sz, asn, doses, std_d_doses, pi_jk, y, K);

    n_t = y.nrow();

    // posterior sampling
    samp = run_sampler_internal(
      N, burn, thin_by, y, n_doses,
      init_a_0, init_a_1, init_a_2, init_a_3,
      init_b_0, init_b_1,
      init_lamb_T, init_lamb_R, init_rho,
      alpha_bar, w, beta_bar, nu,
      eta_T_val, eta_R_val, a_rho, b_rho,
      sig_2, K, Ut_mat, doses, std_d_doses);
    sampler_ran = true;
    
    NumericMatrix exp_ut = samp["exp_ut"];
    int n_samples = exp_ut.nrow();

    // monitoring rule: Pr(EU_l < U_low) against the time-varying cut-off
    NumericVector pr_ubar(n_doses);
    for (int l = 0; l < n_doses; l++) {
      int count = 0;
      for (int i = 0; i < n_samples; i++) if (exp_ut(i, l) < U_low) count++;
      pr_ubar[l] = (double)count / n_samples;
    }

    double cu_t = 1.0 - (double)n_t / N_max * (1.0 - cu_star);

    std::vector<int> unac_v;
    for (int l = 0; l < n_doses; l++) {
      if (pr_ubar[l] > cu_t) unac_v.push_back(l + 1);
    }
    unac_dose_i = IntegerVector(unac_v.size());
    for (size_t u = 0; u < unac_v.size(); u++) unac_dose_i[u] = unac_v[u];

    if ((int)unac_v.size() == n_doses) {

      // no acceptable doses remain
      if (coh_num < max_cohorts) {
        for (int l = 0; l < n_doses; l++) fin_A_t(coh_num, l) = 0;
      }
      stop_reason = "all_unacceptable";
      break;

    } else {

      // A_t = doses[-unac_dose_i]
      std::vector<int> ac_v;
      for (int l = 1; l <= n_doses; l++) {
        bool is_unac = false;
        for (size_t u = 0; u < unac_v.size(); u++) {
          if (unac_v[u] == l) { is_unac = true; break; }
        }
        if (!is_unac) ac_v.push_back(l);
      }
      A_t = IntegerVector(ac_v.size());
      for (size_t a = 0; a < ac_v.size(); a++) A_t[a] = ac_v[a];
    }

    coh_num++;

    // store acceptable dose indicators for this cohort
    if (coh_num - 1 < max_cohorts) {
      for (int l = 0; l < n_doses; l++) {
        bool ac = false;
        for (int a = 0; a < A_t.size(); a++) if (A_t[a] == l + 1) { ac = true; break; }
        fin_A_t(coh_num - 1, l) = ac ? 1 : 0;
      }
    }

    if (n_t >= N_max) break;
  }

  // ---- trial summaries from the final posterior sample ----
  NumericVector post_mean_utility(n_doses + 1, NA_REAL);
  double final_optimal = NA_REAL;
  int final_optimal_i = NA_INTEGER;
  IntegerVector A_t_Count(n_doses);
  
  if (sampler_ran) {
    
    NumericMatrix exp_ut = samp["exp_ut"];
    int n_samples = exp_ut.nrow();
    
    NumericVector mean_eu(n_doses);
    for (int l = 0; l < n_doses; l++) {
      double s = 0.0;
      for (int i = 0; i < n_samples; i++) s += exp_ut(i, l);
      mean_eu[l] = s / n_samples;
    }
    
    // mean EU per dose, then the maximising dose in mg
    for (int l = 0; l < n_doses; l++) post_mean_utility[l] = mean_eu[l];
    
    int imeu = which_max_finite(mean_eu);
    post_mean_utility[n_doses] = (imeu < 0) ? NA_REAL : doses[imeu];
    
    
    // final optimal dose: the acceptable dose with the highest posterior mean
    // expected utility. The acceptable set is the complement of unac_dose_i at
    // the final interim analysis. If every dose is unacceptable there is no
    // optimal dose and final_optimal is left as NA.
    int best = -1;
    for (int l = 0; l < n_doses; l++) {

      bool is_unac = false;
      for (int u = 0; u < unac_dose_i.size(); u++) {
        if (unac_dose_i[u] == l + 1) { is_unac = true; break; }
      }
      if (is_unac) continue;

      // strict > scanning upwards means ties go to the lowest dose,
      // matching which.max
      if (best < 0 || mean_eu[l] > mean_eu[best]) best = l;
    }

    if (best >= 0) {
      final_optimal   = doses[best];
      final_optimal_i = best + 1;   // 1-indexed, for use as an R index
    }
  }
  
  // number of cohorts at which each dose was acceptable (NA rows skipped)
  for (int l = 0; l < n_doses; l++) {
    int c = 0;
    for (int r = 0; r < max_cohorts; r++) {
      if (fin_A_t(r, l) != NA_LOGICAL && fin_A_t(r, l) == 1) c++;
    }
    A_t_Count[l] = c;
  }
  
  return List::create(
    Rcpp::Named("y")                 = y,
    Rcpp::Named("samp")              = samp,
    Rcpp::Named("fin_A_t")           = fin_A_t,
    Rcpp::Named("A_t_Count")         = A_t_Count,
    Rcpp::Named("post_mean_utility") = post_mean_utility,
    Rcpp::Named("final_optimal")     = final_optimal,
    Rcpp::Named("final_optimal_i")   = final_optimal_i,
    Rcpp::Named("unac_dose_i")       = unac_dose_i,
    Rcpp::Named("urn")               = urn,
    Rcpp::Named("trial_end")         = n_t,
    Rcpp::Named("coh_num")           = coh_num,
    Rcpp::Named("stop_reason")       = stop_reason,
    Rcpp::Named("dose_count")        = dose_count);
}

/////////////////////// no_skip_version2 assignment ///////////////////////////

// Internal only - no_skip_version2 dose assignment.
//
// A_t     : 1-indexed acceptable dose positions, increasing
// urn     : modified in place
// y       : current observation matrix (column 2 = Dose, in mg); ignored on
//           the first cohort
// doses   : dose values in mg, increasing, same order as A_t's indices
// n_balls : number to increment urn after each assignment
// coh_num : number of cohorts already completed
//
// Returns the 1-indexed assigned dose, or:
//   -1  if A_t is empty (should not occur - the trial loop stops before
//       calling this once all doses are unacceptable)
//   -2  if A_t is non-empty but no dose at or below d_M(t) is acceptable
//       ("no assignable doses" - a distinct stopping condition from -1)
//   -3  if coh_num > 0 but y is empty (defensive; should not occur)

int no_skip_v2_assign_internal(
    IntegerVector A_t,
    IntegerVector& urn,
    NumericMatrix y,
    NumericVector doses,
    int n_doses,
    int n_balls,
    int coh_num)
{
  int assign;
  
  if (coh_num == 0) {
    
    // step 1: first cohort at d_2
    assign = 2;
    
  } else {
    
    int n_A = A_t.size();
    if (n_A == 0) return -1;
    
    // d_M(t): the maximum dose tried so far, read from y's Dose column.
    // y is guaranteed non-empty whenever coh_num > 0, since coh_num is only
    // incremented after a cohort has been simulated, but guard regardless.
    int n_rows = y.nrow();
    if (n_rows == 0) return -3;
    
    double d_Mt = y(0, 2);
    for (int i = 1; i < n_rows; i++) if (y(i, 2) > d_Mt) d_Mt = y(i, 2);
    
    int d_Mt_ind = -1;
    for (int l = 0; l < n_doses; l++) {
      if (std::abs(doses[l] - d_Mt) < 1e-10) { d_Mt_ind = l + 1; break; }
    }
    
    // no match means the Dose column and the doses vector disagree, which
    // would silently corrupt the escalation rule - fail loudly instead
    if (d_Mt_ind < 0) {
      Rcpp::stop("Maximum tried dose not found in doses; check that doses matches the Dose column of y.");
    }
    
    IntegerVector assignable;
    
    if (d_Mt_ind == n_doses) {
      
      // every dose has been tried: A(t) = A_t
      assignable = A_t;
      
    } else {
      
      int d_Mt_1_idx = d_Mt_ind + 1;   // 1-indexed position of the next dose
      
      bool next_is_acceptable = false;
      for (int i = 0; i < n_A; i++) {
        if (A_t[i] == d_Mt_1_idx) { next_is_acceptable = true; break; }
      }
      
      if (next_is_acceptable) {
        // step 3: escalate to d_{M(t)+1} only, never further
        assignable = IntegerVector::create(d_Mt_1_idx);
      } else {
        // step 3a: A(t) = doses in A_t strictly below d_{M(t)+1}
        std::vector<int> av;
        for (int i = 0; i < n_A; i++) {
          if (A_t[i] < d_Mt_1_idx) av.push_back(A_t[i]);
        }
        assignable = IntegerVector(av.size());
        for (size_t i = 0; i < av.size(); i++) assignable[i] = av[i];
      }
    }
    
    if (assignable.size() == 0) return -2;
    
    if (assignable.size() > 1) {
      NumericVector red_urn(assignable.size());
      for (int i = 0; i < assignable.size(); i++) red_urn[i] = urn[assignable[i] - 1];
      assign = sample_with_prob_internal(assignable, red_urn);
    } else {
      assign = assignable[0];
    }
  }
  
  // urn[-d_ind] <- urn[-d_ind] + 1
  for (int l = 1; l <= n_doses; l++) {
    if (l != assign) urn[l - 1] += n_balls;
  }
  
  return assign;
}


/////////////////////// trial: no_skip_version2 + dichotomised monitoring /////

// One replicate of the adaptive trial using no_skip_version2 assignment and the
// dichotomised toxicity/futility monitoring rule.
//
// Additional arguments relative to the U-Bayes trial:
//   h_T         - toxicity dichotomisation index; dose is judged on
//                 xi_T = Pr(y_T >= h_T), i.e. marginal columns h_T..K[0]-1
//   h_R         - response dichotomisation index; xi_R = Pr(y_R >= h_R)
//   upper_xi_T  - toxicity threshold; dose unacceptable if
//                 Pr(xi_T > upper_xi_T | data) > c_T
//   lower_xi_R  - efficacy threshold; dose unacceptable if
//                 Pr(xi_R < lower_xi_R | data) > c_R

// [[Rcpp::export]]
List run_trial_mpbr(
    int co_sz,
    int n_doses,
    NumericVector doses,
    NumericVector std_d_doses,
    List pi_jk,
    int N_max,
    int n_balls,
    double cu_star_T,
    double cu_star_R,
    int h_T,
    int h_R,
    double upper_xi_T,
    double lower_xi_R,
    int N, int burn, int thin_by,
    NumericVector alpha_bar, NumericVector w,
    NumericVector beta_bar, NumericVector nu,
    Rcpp::Nullable<double> eta_T, Rcpp::Nullable<double> eta_R,
    double a_rho, double b_rho,
    double sig_2,
    IntegerVector K,
    NumericMatrix Ut_mat,
    Rcpp::Nullable<Rcpp::List> init_values = R_NilValue)
{
  if (h_T < 1 || h_T > K[0] - 1) {
    Rcpp::stop("h_T must be between 1 and K[1] - 1.");
  }
  if (h_R < 1 || h_R > K[1] - 1) {
    Rcpp::stop("h_R must be between 1 and K[2] - 1.");
  }
  
  // eta placeholders are unused when the corresponding K = 2
  double eta_T_val = eta_T.isNotNull() ? Rcpp::as<double>(eta_T) : 1.0;
  double eta_R_val = eta_R.isNotNull() ? Rcpp::as<double>(eta_R) : 1.0;
  
  // initial values: defaults, overridden by any supplied in init_values
  double init_a_0 = 0.0, init_a_1 = 1.0, init_a_2 = 0.0, init_a_3 = 1.0;
  double init_b_0 = 0.0, init_b_1 = 1.0;
  double init_rho = 0.0;
  NumericVector init_lamb_T(std::max(0, K[0] - 2), 1.0);
  NumericVector init_lamb_R(std::max(0, K[1] - 2), 1.0);
  
  if (init_values.isNotNull()) {
    Rcpp::List iv(init_values);
    if (iv.containsElementNamed("cur_a_0")) init_a_0 = Rcpp::as<double>(iv["cur_a_0"]);
    if (iv.containsElementNamed("cur_a_1")) init_a_1 = Rcpp::as<double>(iv["cur_a_1"]);
    if (iv.containsElementNamed("cur_a_2")) init_a_2 = Rcpp::as<double>(iv["cur_a_2"]);
    if (iv.containsElementNamed("cur_a_3")) init_a_3 = Rcpp::as<double>(iv["cur_a_3"]);
    if (iv.containsElementNamed("cur_b_0")) init_b_0 = Rcpp::as<double>(iv["cur_b_0"]);
    if (iv.containsElementNamed("cur_b_1")) init_b_1 = Rcpp::as<double>(iv["cur_b_1"]);
    if (iv.containsElementNamed("cur_rho")) init_rho = Rcpp::as<double>(iv["cur_rho"]);
    if (iv.containsElementNamed("cur_lamb_T")) {
      SEXP v = iv["cur_lamb_T"];
      init_lamb_T = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
    if (iv.containsElementNamed("cur_lamb_R")) {
      SEXP v = iv["cur_lamb_R"];
      init_lamb_R = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
  }
  
  // initialise trial state
  int n_t = 0;
  int coh_num = 0;
  
  IntegerVector dose_count(n_doses);
  
  IntegerVector urn(n_doses);
  for (int l = 0; l < n_doses; l++) urn[l] = 1;
  
  NumericMatrix y(0, 4);
  Rcpp::colnames(y) = Rcpp::CharacterVector::create(
    "y_T", "y_R", "Dose", "std_d");
  
  int max_cohorts = (int)std::ceil((double)N_max / co_sz);
  Rcpp::LogicalMatrix fin_A_t(max_cohorts, n_doses);
  std::fill(fin_A_t.begin(), fin_A_t.end(), NA_LOGICAL);
  
  // A_t as 1-indexed dose positions; starts as all doses
  IntegerVector A_t(n_doses);
  for (int l = 0; l < n_doses; l++) A_t[l] = l + 1;
  
  IntegerVector unac_dose_i(0);
  NumericVector pr_xi_T(n_doses), pr_xi_R(n_doses);
  List samp;
  bool sampler_ran = false;
  
  // "max_reached", "all_unacceptable", or "no_assignable"
  Rcpp::String stop_reason = "max_reached";
  
  while (true) {
    
    // assign a dose, simulate the cohort, update the urn
    int asn = no_skip_v2_assign_internal(A_t, urn, y, doses, n_doses, n_balls,
                                         coh_num);
    
    if (asn == -2) { stop_reason = "no_assignable"; break; }
    if (asn == -3) { Rcpp::stop("Internal error: empty y with coh_num > 0."); }
    if (asn < 0)   { stop_reason = "all_unacceptable"; break; }
    
    dose_count[asn - 1]++;
    
    y = sim_cohort_internal(co_sz, asn, doses, std_d_doses, pi_jk, y, K);
    
    n_t = y.nrow();
    
    // posterior sampling
    samp = run_sampler_internal(
      N, burn, thin_by, y, n_doses,
      init_a_0, init_a_1, init_a_2, init_a_3,
      init_b_0, init_b_1,
      init_lamb_T, init_lamb_R, init_rho,
      alpha_bar, w, beta_bar, nu,
      eta_T_val, eta_R_val, a_rho, b_rho,
      sig_2, K, Ut_mat, doses, std_d_doses);
    sampler_ran = true;
    
    // ---- monitoring rule on the dichotomised marginals ----
    // For each retained draw and each dose:
    //   xi_T = sum over toxicity categories h_T..K[0]-1 of the marginal
    //   xi_R = sum over response categories h_R..K[1]-1 of the marginal
    // These are row/column sums of the joint, so they are taken directly from
    // post_pi_jk rather than forming the full marginal matrices.
    List post_pi = samp["post_pi_jk"];
    int n_samples = post_pi.size();
    
    std::vector<int> cnt_T(n_doses, 0), cnt_R(n_doses, 0);
    
    for (int i = 0; i < n_samples; i++) {
      
      List pi_draw = post_pi[i];
      
      for (int l = 0; l < n_doses; l++) {
        
        NumericMatrix pm = pi_draw[l];
        
        double xi_T = 0.0;
        for (int t = h_T; t < K[0]; t++) {
          for (int r = 0; r < K[1]; r++) xi_T += pm(t, r);
        }
        
        double xi_R = 0.0;
        for (int r = h_R; r < K[1]; r++) {
          for (int t = 0; t < K[0]; t++) xi_R += pm(t, r);
        }
        
        if (xi_T > upper_xi_T) cnt_T[l]++;
        if (xi_R < lower_xi_R) cnt_R[l]++;
      }
    }
    
    for (int l = 0; l < n_doses; l++) {
      pr_xi_T[l] = (double)cnt_T[l] / n_samples;
      pr_xi_R[l] = (double)cnt_R[l] / n_samples;
    }
    
    // time-varying cut-offs (identical for toxicity and futility)
    double c_T = 1.0 - (double)n_t / N_max * (1.0 - cu_star_T);
    double c_R = 1.0 - (double)n_t / N_max * (1.0 - cu_star_R);
    
    // a dose is unacceptable if it fails either criterion
    std::vector<int> unac_v;
    for (int l = 0; l < n_doses; l++) {
      if (pr_xi_T[l] > c_T || pr_xi_R[l] > c_R) unac_v.push_back(l + 1);
    }
    unac_dose_i = IntegerVector(unac_v.size());
    for (size_t u = 0; u < unac_v.size(); u++) unac_dose_i[u] = unac_v[u];
    
    if ((int)unac_v.size() == n_doses) {
      
      if (coh_num < max_cohorts) {
        for (int l = 0; l < n_doses; l++) fin_A_t(coh_num, l) = 0;
      }
      stop_reason = "all_unacceptable";
      break;
      
    } else {
      
      std::vector<int> ac_v;
      for (int l = 1; l <= n_doses; l++) {
        bool is_unac = false;
        for (size_t u = 0; u < unac_v.size(); u++) {
          if (unac_v[u] == l) { is_unac = true; break; }
        }
        if (!is_unac) ac_v.push_back(l);
      }
      A_t = IntegerVector(ac_v.size());
      for (size_t a = 0; a < ac_v.size(); a++) A_t[a] = ac_v[a];
    }
    
    coh_num++;
    
    if (coh_num - 1 < max_cohorts) {
      for (int l = 0; l < n_doses; l++) {
        bool ac = false;
        for (int a = 0; a < A_t.size(); a++) if (A_t[a] == l + 1) { ac = true; break; }
        fin_A_t(coh_num - 1, l) = ac ? 1 : 0;
      }
    }
    
    if (n_t >= N_max) break;
  }
  
  // ---- trial summaries from the final posterior sample ----
  // (skipped if the sampler never ran, i.e. no_assignable fired on the very
  // first cohort - not reachable given coh_num == 0 always assigns dose 2,
  // but guarded here rather than assumed) 
  // aka does not have equivalent behavior in R since R does not have a fail safe
  
  NumericVector post_mean_utility(n_doses + 1, NA_REAL);
  double final_optimal = NA_REAL;
  int final_optimal_i = NA_INTEGER;
  IntegerVector A_t_Count(n_doses);
  
  if (sampler_ran) {
    
    NumericMatrix exp_ut = samp["exp_ut"];
    int n_samples = exp_ut.nrow();
    
    NumericVector mean_eu(n_doses);
    for (int l = 0; l < n_doses; l++) {
      double s = 0.0;
      for (int i = 0; i < n_samples; i++) s += exp_ut(i, l);
      mean_eu[l] = s / n_samples;
    }
    
    for (int l = 0; l < n_doses; l++) post_mean_utility[l] = mean_eu[l];
    
    int im = which_max_finite(mean_eu);
    post_mean_utility[n_doses] = (im < 0) ? NA_REAL : doses[im];
    
    // build set of tried dose indices (0-indexed)
    std::vector<bool> tried(n_doses, false);
    for (int i = 0; i < y.nrow(); i++) {
      for (int l = 0; l < n_doses; l++) {
        if (std::abs(y(i, 2) - doses[l]) < 1e-10) { tried[l] = true; break; }
      }
    }
    
    // final optimal: must be both acceptable AND tried
    int best = -1;
    for (int l = 0; l < n_doses; l++) {
      
      if (!tried[l]) continue;
      
      bool is_unac = false;
      for (int u = 0; u < unac_dose_i.size(); u++) {
        if (unac_dose_i[u] == l + 1) { is_unac = true; break; }
      }
      if (is_unac) continue;
      
      if (best < 0 || mean_eu[l] > mean_eu[best]) best = l;
    }
    
    if (best >= 0) {
      final_optimal   = doses[best];
      final_optimal_i = best + 1;
    }
  }
  // number of cohorts at which each dose was acceptable (NA rows skipped)
  for (int l = 0; l < n_doses; l++) {
    int c = 0;
    for (int r = 0; r < max_cohorts; r++) {
      if (fin_A_t(r, l) != NA_LOGICAL && fin_A_t(r, l) == 1) c++;
    }
    A_t_Count[l] = c;
  }
  
  return List::create(
    Rcpp::Named("y")                 = y,
    Rcpp::Named("samp")              = samp,
    Rcpp::Named("fin_A_t")           = fin_A_t,
    Rcpp::Named("A_t_Count")         = A_t_Count,
    Rcpp::Named("post_mean_utility") = post_mean_utility,
    Rcpp::Named("final_optimal")     = final_optimal,
    Rcpp::Named("final_optimal_i")   = final_optimal_i,
    Rcpp::Named("pr_xi_T")           = pr_xi_T,
    Rcpp::Named("pr_xi_R")           = pr_xi_R,
    Rcpp::Named("unac_dose_i")       = unac_dose_i,
    Rcpp::Named("urn")               = urn,
    Rcpp::Named("trial_end")         = n_t,
    Rcpp::Named("coh_num")           = coh_num,
    Rcpp::Named("stop_reason")       = stop_reason,
    Rcpp::Named("dose_count")         = dose_count);
}


/////////// trial: no_skip_version2, toxicity monitoring only while small //////

// One replicate of the adaptive trial using no_skip_version2 assignment, where
// marginal toxicity contributes to dose removal only while the accrued sample
// size is at or below veto_point. Beyond that point doses are removed on
// expected utility alone.
//
// Requires no_skip_v2_assign_internal, sim_cohort_internal and
// run_sampler_internal, all defined earlier in this file.
//
// Monitoring arguments
//   h_T        - toxicity dichotomisation index; xi_T = Pr(y_T >= h_T),
//                i.e. marginal toxicity categories h_T..K[0]-1
//   upper_xi_T - toxicity threshold; a dose is unacceptable if
//                Pr(xi_T > upper_xi_T | data) > c_T
//   U_low      - utility lower bound; a dose is unacceptable if
//                Pr(EU < U_low | data) > cu_t
//   cu_star_T  - final toxicity cut-off, used to form c_T
//   cu_star    - final utility cut-off, used to form cu_t
//   veto_point  - largest n_t at which the toxicity criterion still applies;
//                for n_t > veto_point only the utility criterion is used
//
// Both cut-offs shrink with accrual: c = 1 - n_t/N_max * (1 - cu_star).
// [[Rcpp::export]]
List run_trial_veto_ubr(
    int co_sz,
    int n_doses,
    NumericVector doses,
    NumericVector std_d_doses,
    List pi_jk,
    int N_max,
    int n_balls,
    double cu_star_T,
    double cu_star,
    int h_T,
    double upper_xi_T,
    double U_low,
    int veto_point,
    int N, int burn, int thin_by,
    NumericVector alpha_bar, NumericVector w,
    NumericVector beta_bar, NumericVector nu,
    Rcpp::Nullable<double> eta_T, Rcpp::Nullable<double> eta_R,
    double a_rho, double b_rho,
    double sig_2,
    IntegerVector K,
    NumericMatrix Ut_mat,
    Rcpp::Nullable<Rcpp::List> init_values = R_NilValue)
{
  if (h_T < 1 || h_T > K[0] - 1) {
    Rcpp::stop("h_T must be between 1 and K[1] - 1.");
  }
  
  // the toxicity criterion is only ever evaluated after a cohort has accrued,
  // so it can never apply if the first cohort already exceeds veto_point
  if (co_sz > veto_point) {
    Rcpp::warning("co_sz exceeds veto_point, so the toxicity criterion will never be applied.");
  }
  
  // eta placeholders are unused when the corresponding K = 2
  double eta_T_val = eta_T.isNotNull() ? Rcpp::as<double>(eta_T) : 1.0;
  double eta_R_val = eta_R.isNotNull() ? Rcpp::as<double>(eta_R) : 1.0;
  
  // initial values: defaults, overridden by any supplied in init_values
  double init_a_0 = 0.0, init_a_1 = 1.0, init_a_2 = 0.0, init_a_3 = 1.0;
  double init_b_0 = 0.0, init_b_1 = 1.0;
  double init_rho = 0.0;
  NumericVector init_lamb_T(std::max(0, K[0] - 2), 1.0);
  NumericVector init_lamb_R(std::max(0, K[1] - 2), 1.0);
  
  if (init_values.isNotNull()) {
    Rcpp::List iv(init_values);
    if (iv.containsElementNamed("cur_a_0")) init_a_0 = Rcpp::as<double>(iv["cur_a_0"]);
    if (iv.containsElementNamed("cur_a_1")) init_a_1 = Rcpp::as<double>(iv["cur_a_1"]);
    if (iv.containsElementNamed("cur_a_2")) init_a_2 = Rcpp::as<double>(iv["cur_a_2"]);
    if (iv.containsElementNamed("cur_a_3")) init_a_3 = Rcpp::as<double>(iv["cur_a_3"]);
    if (iv.containsElementNamed("cur_b_0")) init_b_0 = Rcpp::as<double>(iv["cur_b_0"]);
    if (iv.containsElementNamed("cur_b_1")) init_b_1 = Rcpp::as<double>(iv["cur_b_1"]);
    if (iv.containsElementNamed("cur_rho")) init_rho = Rcpp::as<double>(iv["cur_rho"]);
    if (iv.containsElementNamed("cur_lamb_T")) {
      SEXP v = iv["cur_lamb_T"];
      init_lamb_T = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
    if (iv.containsElementNamed("cur_lamb_R")) {
      SEXP v = iv["cur_lamb_R"];
      init_lamb_R = (v == R_NilValue) ? NumericVector(0) : Rcpp::as<NumericVector>(v);
    }
  }
  
  // initialise trial state
  int n_t = 0;
  int coh_num = 0;
  
  IntegerVector dose_count(n_doses);
  
  IntegerVector urn(n_doses);
  for (int l = 0; l < n_doses; l++) urn[l] = 1;
  
  NumericMatrix y(0, 4);
  Rcpp::colnames(y) = Rcpp::CharacterVector::create(
    "y_T", "y_R", "Dose", "std_d");
  
  int max_cohorts = (int)std::ceil((double)N_max / co_sz);
  Rcpp::LogicalMatrix fin_A_t(max_cohorts, n_doses);
  std::fill(fin_A_t.begin(), fin_A_t.end(), NA_LOGICAL);
  
  // A_t as 1-indexed dose positions; starts as all doses
  IntegerVector A_t(n_doses);
  for (int l = 0; l < n_doses; l++) A_t[l] = l + 1;
  
  IntegerVector unac_dose_i(0);
  // pr_xi_T stays NA at interim analyses where toxicity is not monitored
  NumericVector pr_xi_T(n_doses, NA_REAL), pr_ubar(n_doses, NA_REAL);
  List samp;
  bool sampler_ran = false;
  bool tox_applied = false;
  
  // "max_reached", "all_unacceptable", or "no_assignable"
  Rcpp::String stop_reason = "max_reached";
  
  while (true) {
    
    // assign a dose, simulate the cohort, update the urn
    int asn = no_skip_v2_assign_internal(A_t, urn, y, doses, n_doses, n_balls, 
                                         coh_num);
    
    if (asn == -2) { stop_reason = "no_assignable"; break; }
    if (asn == -3) { Rcpp::stop("Internal error: empty y with coh_num > 0."); }
    if (asn < 0)   { stop_reason = "all_unacceptable"; break; }
    
    dose_count[asn - 1]++;
    
    y = sim_cohort_internal(co_sz, asn, doses, std_d_doses, pi_jk, y, K);
    
    n_t = y.nrow();
    
    // posterior sampling
    samp = run_sampler_internal(
      N, burn, thin_by, y, n_doses,
      init_a_0, init_a_1, init_a_2, init_a_3,
      init_b_0, init_b_1,
      init_lamb_T, init_lamb_R, init_rho,
      alpha_bar, w, beta_bar, nu,
      eta_T_val, eta_R_val, a_rho, b_rho,
      sig_2, K, Ut_mat, doses, std_d_doses);
    sampler_ran = true;
    
    NumericMatrix exp_ut = samp["exp_ut"];
    int n_eu = exp_ut.nrow();
    
    // ---- utility criterion: applied at every interim analysis ----
    for (int l = 0; l < n_doses; l++) {
      int c = 0;
      for (int i = 0; i < n_eu; i++) if (exp_ut(i, l) < U_low) c++;
      pr_ubar[l] = (double)c / n_eu;
    }
    
    double cu_t = 1.0 - (double)n_t / N_max * (1.0 - cu_star);
    
    // ---- toxicity criterion: only while n_t <= veto_point ----
    bool use_tox = (n_t <= veto_point);
    double c_T = 1.0 - (double)n_t / N_max * (1.0 - cu_star_T);
    
    if (use_tox) {
      
      tox_applied = true;
      
      List post_pi = samp["post_pi_jk"];
      int n_samples = post_pi.size();
      
      std::vector<int> cnt_T(n_doses, 0);
      
      for (int i = 0; i < n_samples; i++) {
        
        List pi_draw = post_pi[i];
        
        for (int l = 0; l < n_doses; l++) {
          
          NumericMatrix pm = pi_draw[l];
          
          double xi_T = 0.0;
          for (int t = h_T; t < K[0]; t++) {
            for (int r = 0; r < K[1]; r++) xi_T += pm(t, r);
          }
          
          if (xi_T > upper_xi_T) cnt_T[l]++;
        }
      }
      
      for (int l = 0; l < n_doses; l++) {
        pr_xi_T[l] = (double)cnt_T[l] / n_samples;
      }
      
    } else {
      
      // not monitored at this analysis
      for (int l = 0; l < n_doses; l++) pr_xi_T[l] = NA_REAL;
    }
    
    // a dose is removed if it fails either active criterion
    std::vector<int> unac_v;
    for (int l = 0; l < n_doses; l++) {
      bool bad = (pr_ubar[l] > cu_t);
      if (use_tox && pr_xi_T[l] > c_T) bad = true;
      if (bad) unac_v.push_back(l + 1);
    }
    unac_dose_i = IntegerVector(unac_v.size());
    for (size_t u = 0; u < unac_v.size(); u++) unac_dose_i[u] = unac_v[u];
    
    if ((int)unac_v.size() == n_doses) {
      
      if (coh_num < max_cohorts) {
        for (int l = 0; l < n_doses; l++) fin_A_t(coh_num, l) = 0;
      }
      stop_reason = "all_unacceptable";
      break;
      
    } else {
      
      std::vector<int> ac_v;
      for (int l = 1; l <= n_doses; l++) {
        bool is_unac = false;
        for (size_t u = 0; u < unac_v.size(); u++) {
          if (unac_v[u] == l) { is_unac = true; break; }
        }
        if (!is_unac) ac_v.push_back(l);
      }
      A_t = IntegerVector(ac_v.size());
      for (size_t a = 0; a < ac_v.size(); a++) A_t[a] = ac_v[a];
    }
    
    coh_num++;
    
    if (coh_num - 1 < max_cohorts) {
      for (int l = 0; l < n_doses; l++) {
        bool ac = false;
        for (int a = 0; a < A_t.size(); a++) if (A_t[a] == l + 1) { ac = true; break; }
        fin_A_t(coh_num - 1, l) = ac ? 1 : 0;
      }
    }
    
    if (n_t >= N_max) break;
  }
  
  // ---- trial summaries from the final posterior sample ----
  NumericVector post_mean_utility(n_doses + 1, NA_REAL);
  double final_optimal = NA_REAL;
  int final_optimal_i = NA_INTEGER;
  IntegerVector A_t_Count(n_doses);
  
  if (sampler_ran) {
    
    NumericMatrix exp_ut = samp["exp_ut"];
    int n_samples = exp_ut.nrow();
    
    NumericVector mean_eu(n_doses);
    for (int l = 0; l < n_doses; l++) {
      double s = 0.0;
      for (int i = 0; i < n_samples; i++) s += exp_ut(i, l);
      mean_eu[l] = s / n_samples;
    }
    
    for (int l = 0; l < n_doses; l++) post_mean_utility[l] = mean_eu[l];
    
    int im = which_max_finite(mean_eu);
    post_mean_utility[n_doses] = (im < 0) ? NA_REAL : doses[im];
    
    // build set of tried dose indices (0-indexed)
    std::vector<bool> tried(n_doses, false);
    for (int i = 0; i < y.nrow(); i++) {
      for (int l = 0; l < n_doses; l++) {
        if (std::abs(y(i, 2) - doses[l]) < 1e-10) { tried[l] = true; break; }
      }
    }
    
    // final optimal: must be both acceptable AND tried
    int best = -1;
    for (int l = 0; l < n_doses; l++) {
      
      if (!tried[l]) continue;
      
      bool is_unac = false;
      for (int u = 0; u < unac_dose_i.size(); u++) {
        if (unac_dose_i[u] == l + 1) { is_unac = true; break; }
      }
      if (is_unac) continue;
      
      if (best < 0 || mean_eu[l] > mean_eu[best]) best = l;
    }
    
    if (best >= 0) {
      final_optimal   = doses[best];
      final_optimal_i = best + 1;
    }
  }
  // number of cohorts at which each dose was acceptable (NA rows skipped)
  for (int l = 0; l < n_doses; l++) {
    int c = 0;
    for (int r = 0; r < max_cohorts; r++) {
      if (fin_A_t(r, l) != NA_LOGICAL && fin_A_t(r, l) == 1) c++;
    }
    A_t_Count[l] = c;
  }
  
  return List::create(
    Rcpp::Named("y")                 = y,
    Rcpp::Named("samp")              = samp,
    Rcpp::Named("fin_A_t")           = fin_A_t,
    Rcpp::Named("A_t_Count")         = A_t_Count,
    Rcpp::Named("post_mean_utility") = post_mean_utility,
    Rcpp::Named("final_optimal")     = final_optimal,
    Rcpp::Named("final_optimal_i")   = final_optimal_i,
    Rcpp::Named("pr_xi_T")           = pr_xi_T,
    Rcpp::Named("pr_ubar")           = pr_ubar,
    Rcpp::Named("tox_applied")       = tox_applied,
    Rcpp::Named("unac_dose_i")       = unac_dose_i,
    Rcpp::Named("urn")               = urn,
    Rcpp::Named("trial_end")         = n_t,
    Rcpp::Named("coh_num")           = coh_num,
    Rcpp::Named("stop_reason")       = stop_reason,
    Rcpp::Named("dose_count")        = dose_count);
}