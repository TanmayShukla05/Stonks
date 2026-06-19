#include "httplib.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
using namespace std;

// ===================== CONFIG =====================
const int NUM_STOCKS = 12;
const double BASE_A = 0.02;     // 2% base volatility 'a'
const int HISTORY_LEN = 50;     // rolling window for correlation + signal estimation
const int MCMC_ITERATIONS = 1000;  // MCMC sampling iterations
const int MCMC_BURNIN = 200;       // burn-in period

vector<double> prices(NUM_STOCKS, 100.0);
vector<string> modes(NUM_STOCKS, "normal");      // normal, crash, boost (true peaks, lasts 1 tick)
vector<vector<double>> true_correlations(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0)); // ground truth, hidden from frontend
vector<vector<double>> returns_history;           // per-stock rolling window of realized returns, used by the estimator

mt19937 rng(1337);

void init_data() {
    uniform_real_distribution<double> corr_dist(0.0, 0.5);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        true_correlations[i][i] = 1.0;
        for (int j = i + 1; j < NUM_STOCKS; ++j) {
            double c = corr_dist(rng);
            true_correlations[i][j] = true_correlations[j][i] = c;
        }
        returns_history.push_back(vector<double>());
    }
}

// True bimodal distribution: a 50/50 mixture of two normals centered at peakLow/peakHigh.
// normal mode  -> peaks at {-a, +a}
// crash mode   -> peaks at {-6a, -4a}   (shifted hard negative, still two-humped)
// boost mode   -> peaks at {+4a, +6a}   (shifted hard positive, still two-humped)
double get_bimodal_return(const string& mode) {
    double peakLow, peakHigh;
    if (mode == "crash") { peakLow = -6 * BASE_A; peakHigh = -4 * BASE_A; }
    else if (mode == "boost") { peakLow = 4 * BASE_A; peakHigh = 6 * BASE_A; }
    else { peakLow = -BASE_A; peakHigh = BASE_A; }

    uniform_real_distribution<double> coin(0.0, 1.0);
    double peak = (coin(rng) < 0.5) ? peakLow : peakHigh;
    normal_distribution<double> dist(peak, BASE_A * 0.25); // tight bump around each peak
    return dist(rng);
}

void tick_simulation() {
    vector<double> current_returns(NUM_STOCKS);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        current_returns[i] = get_bimodal_return(modes[i]);
        if (modes[i] != "normal") modes[i] = "normal"; // disrupt/enhance only lasts 1 tick
    }

    // Apply correlated impact: each stock is nudged by a correlation-weighted sum
    // of *all* shocks this tick (including its own), scaled down so correlation
    // effects are felt but don't dominate the stock's own move.
    for (int i = 0; i < NUM_STOCKS; ++i) {
        double total_shock = 0.0;
        for (int j = 0; j < NUM_STOCKS; ++j) {
            double weight = (i == j) ? 1.0 : true_correlations[i][j] * 0.2;
            total_shock += current_returns[j] * weight;
        }
        prices[i] *= (1.0 + total_shock);

        // Store the REALIZED return (post-blend), not the pre-blend independent draw.
        // The realized return is what actually carries the correlation signal between
        // stocks -- using the pre-blend draw here would make every stock's history
        // independent by construction, and the estimator could never recover the
        // true correlations no matter how long it ran.
        returns_history[i].push_back(total_shock);
        if ((int)returns_history[i].size() > HISTORY_LEN) returns_history[i].erase(returns_history[i].begin());
    }
}

// Ensure correlation matrix is valid (symmetric, positive semi-definite diagonal=1)
void project_to_correlation_matrix(vector<vector<double>>& corr) {
    int n = corr.size();
    // Ensure symmetry
    for (int i = 0; i < n; ++i) {
        corr[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double avg = (corr[i][j] + corr[j][i]) / 2.0;
            avg = max(-0.99, min(0.99, avg)); // clamp to valid correlation range
            corr[i][j] = corr[j][i] = avg;
        }
    }
}

// Log-likelihood of observed returns given a correlation matrix
// Using multivariate normal assumption (simplified)
double log_likelihood(const vector<vector<double>>& corr, const vector<vector<double>>& returns) {
    int n = returns.size();
    int T = returns[0].size();
    if (T < 2) return 0.0;
    
    double ll = 0.0;
    
    // For each time point, compute likelihood
    for (int t = 0; t < T; ++t) {
        vector<double> r(n);
        for (int i = 0; i < n; ++i) {
            if (t < (int)returns[i].size()) r[i] = returns[i][t];
            else r[i] = 0.0;
        }
        
        // Simplified: sum of squared standardized residuals weighted by correlation
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    ll -= 0.5 * r[i] * r[i] / (BASE_A * BASE_A);
                } else {
                    // Correlation reduces the penalty when returns move together
                    ll -= 0.5 * (1.0 - corr[i][j]) * r[i] * r[j] / (BASE_A * BASE_A);
                }
            }
        }
    }
    
    return ll;
}

// MCMC-based correlation estimation using Metropolis-Hastings
vector<vector<double>> estimate_correlations_mcmc() {
    int n = NUM_STOCKS;
    
    // Check if we have enough data
    bool has_data = false;
    int min_samples = 10;
    for (int i = 0; i < n; ++i) {
        if ((int)returns_history[i].size() >= min_samples) {
            has_data = true;
            break;
        }
    }
    
    if (!has_data) {
        // Return identity matrix if not enough data
        vector<vector<double>> identity(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) identity[i][i] = 1.0;
        return identity;
    }
    
    // Initialize with simple Pearson correlation as starting point
    vector<vector<double>> current_corr(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        current_corr[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            // Simple correlation estimate
            int len = min(returns_history[i].size(), returns_history[j].size());
            if (len < 5) {
                current_corr[i][j] = current_corr[j][i] = 0.0;
                continue;
            }
            
            double mi = 0, mj = 0;
            for (int k = 0; k < len; ++k) {
                mi += returns_history[i][k];
                mj += returns_history[j][k];
            }
            mi /= len; mj /= len;
            
            double cov = 0, vi = 0, vj = 0;
            for (int k = 0; k < len; ++k) {
                double di = returns_history[i][k] - mi;
                double dj = returns_history[j][k] - mj;
                cov += di * dj;
                vi += di * di;
                vj += dj * dj;
            }
            
            if (vi > 1e-12 && vj > 1e-12) {
                current_corr[i][j] = current_corr[j][i] = cov / sqrt(vi * vj);
            }
        }
    }
    
    double current_ll = log_likelihood(current_corr, returns_history);
    
    // Accumulator for posterior mean
    vector<vector<double>> mean_corr(n, vector<double>(n, 0.0));
    int samples_collected = 0;
    
    uniform_real_distribution<double> uniform(0.0, 1.0);
    normal_distribution<double> proposal(0.0, 0.05); // proposal step size
    
    // MCMC iterations
    for (int iter = 0; iter < MCMC_ITERATIONS; ++iter) {
        // Propose a new correlation matrix by perturbing one element
        vector<vector<double>> proposed_corr = current_corr;
        
        // Randomly select an off-diagonal element to perturb
        int i = (int)(uniform(rng) * n);
        int j = (int)(uniform(rng) * n);
        if (i == j) {
            j = (j + 1) % n;
        }
        if (i > j) swap(i, j);
        
        // Propose new value
        double new_val = current_corr[i][j] + proposal(rng);
        new_val = max(-0.99, min(0.99, new_val)); // keep in valid range
        
        proposed_corr[i][j] = proposed_corr[j][i] = new_val;
        
        // Compute likelihood of proposed state
        double proposed_ll = log_likelihood(proposed_corr, returns_history);
        
        // Metropolis-Hastings acceptance
        double log_alpha = proposed_ll - current_ll;
        
        if (log(uniform(rng)) < log_alpha) {
            // Accept
            current_corr = proposed_corr;
            current_ll = proposed_ll;
        }
        
        // After burn-in, collect samples
        if (iter >= MCMC_BURNIN) {
            for (int ii = 0; ii < n; ++ii) {
                for (int jj = 0; jj < n; ++jj) {
                    mean_corr[ii][jj] += current_corr[ii][jj];
                }
            }
            samples_collected++;
        }
    }
    
    // Compute posterior mean
    if (samples_collected > 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                mean_corr[i][j] /= samples_collected;
            }
        }
    }
    
    project_to_correlation_matrix(mean_corr);
    return mean_corr;
}

// Correlation-weighted predicted next return for stock i: a weighted average of
// every OTHER stock's most recent realized return, weighted by the model's own
// estimated correlation (not the hidden true one). This is the "factor model"
// signal: if stocks the model believes are correlated to i just moved up, i is
// expected to drift up too.
double predicted_return(int i, const vector<vector<double>>& est) {
    if (returns_history[i].empty()) return 0.0;
    double weighted_sum = 0.0, weight_total = 0.0;
    for (int j = 0; j < NUM_STOCKS; ++j) {
        if (j == i || returns_history[j].empty()) continue;
        double w = fabs(est[i][j]);
        if (w < 1e-6) continue;
        double last_return_j = returns_history[j].back();
        weighted_sum += est[i][j] * last_return_j * w; // sign of est[i][j] matters, weight by |corr|
        weight_total += w;
    }
    if (weight_total < 1e-6) return 0.0;
    return weighted_sum / weight_total;
}

int main() {
    init_data();
    httplib::Server svr;
    svr.set_mount_point("/", "./static");

    svr.Get("/api/tick", [](const httplib::Request&, httplib::Response& res) {
        tick_simulation();
        res.set_content("OK", "text/plain");
    });

    svr.Get("/api/data", [](const httplib::Request&, httplib::Response& res) {
        auto est = estimate_correlations_mcmc();
        stringstream ss;
        ss << "[";
        for (int i = 0; i < NUM_STOCKS; ++i) {
            if (i > 0) ss << ",";
            double pred = predicted_return(i, est);
            // Squash predicted return into a 0-100 "confidence" reading.
            // tanh keeps it bounded and roughly linear for small pred values.
            double signal = tanh(pred / BASE_A); // -1 (strong sell) .. +1 (strong buy)
            string action = (signal >= 0) ? "BUY" : "SELL";
            double confidence = fabs(signal) * 100.0;
            ss << "{\"p\":" << prices[i]
               << ",\"action\":\"" << action << "\""
               << ",\"confidence\":" << confidence << "}";
        }
        ss << "]";
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/api/correlations", [](const httplib::Request&, httplib::Response& res) {
        auto est = estimate_correlations_mcmc();
        double total_abs_error = 0.0;
        int pair_count = 0;
        stringstream ss;
        ss << "{\"error\":[";
        for (int i = 0; i < NUM_STOCKS; ++i) {
            if (i > 0) ss << ",";
            ss << "[";
            for (int j = 0; j < NUM_STOCKS; ++j) {
                if (j > 0) ss << ",";
                double err = est[i][j] - true_correlations[i][j];
                ss << err;
                if (i != j) { total_abs_error += fabs(err); pair_count++; }
            }
            ss << "]";
        }
        ss << "],";
        double mae = (pair_count > 0) ? total_abs_error / pair_count : 0.0;
        ss << "\"mae\":" << mae;
        ss << "}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/api/action", [](const httplib::Request& req, httplib::Response& res) {
        auto id = stoi(req.get_param_value("id"));
        auto type = req.get_param_value("type");
        if (id >= 0 && id < NUM_STOCKS) modes[id] = type;
        res.set_content("OK", "text/plain");
    });

    cout << "Server starting..." << endl;
    svr.listen("0.0.0.0", stoi(getenv("PORT") ? getenv("PORT") : "8080"));
}
