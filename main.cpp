#include "httplib.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
using namespace std;

// ===================== CONFIG =====================
const int NUM_STOCKS = 12;
const double BASE_A = 0.02;
const int HISTORY_LEN = 50;
const double EWMA_LAMBDA = 0.94;

vector<double> prices(NUM_STOCKS, 100.0);
vector<string> modes(NUM_STOCKS, "normal");
vector<vector<double>> true_correlations(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
vector<vector<double>> returns_history;

vector<vector<double>> ewma_cov(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
vector<double> ewma_mean(NUM_STOCKS, 0.0);
bool ewma_initialized = false;
int tick_count = 0;

mt19937 rng(1337);

void init_data() {
    cout << "Initializing data..." << endl;
    uniform_real_distribution<double> corr_dist(0.0, 0.5);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        true_correlations[i][i] = 1.0;
        for (int j = i + 1; j < NUM_STOCKS; ++j) {
            double c = corr_dist(rng);
            true_correlations[i][j] = true_correlations[j][i] = c;
        }
        returns_history.push_back(vector<double>());
    }
    for (int i = 0; i < NUM_STOCKS; ++i) {
        ewma_cov[i][i] = BASE_A * BASE_A;
    }
    cout << "Data initialized successfully" << endl;
}

double get_bimodal_return(const string& mode) {
    double peakLow, peakHigh;
    if (mode == "crash") { peakLow = -6 * BASE_A; peakHigh = -4 * BASE_A; }
    else if (mode == "boost") { peakLow = 4 * BASE_A; peakHigh = 6 * BASE_A; }
    else { peakLow = -BASE_A; peakHigh = BASE_A; }

    uniform_real_distribution<double> coin(0.0, 1.0);
    double peak = (coin(rng) < 0.5) ? peakLow : peakHigh;
    normal_distribution<double> dist(peak, BASE_A * 0.25);
    return dist(rng);
}

void update_ewma(const vector<double>& new_returns) {
    double lam = EWMA_LAMBDA;

    if (!ewma_initialized) {
        for (int i = 0; i < NUM_STOCKS; ++i) ewma_mean[i] = new_returns[i];
        ewma_initialized = true;
        return;
    }

    vector<double> old_mean = ewma_mean;
    for (int i = 0; i < NUM_STOCKS; ++i) {
        ewma_mean[i] = lam * ewma_mean[i] + (1.0 - lam) * new_returns[i];
    }

    vector<double> dev(NUM_STOCKS);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        dev[i] = new_returns[i] - old_mean[i];
    }

    for (int i = 0; i < NUM_STOCKS; ++i) {
        for (int j = i; j < NUM_STOCKS; ++j) {
            ewma_cov[i][j] = lam * ewma_cov[i][j] + (1.0 - lam) * dev[i] * dev[j];
            ewma_cov[j][i] = ewma_cov[i][j];
        }
    }
}

void tick_simulation() {
    vector<double> current_returns(NUM_STOCKS);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        current_returns[i] = get_bimodal_return(modes[i]);
        if (modes[i] != "normal") modes[i] = "normal";
    }

    vector<double> realized_returns(NUM_STOCKS);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        double total_shock = 0.0;
        for (int j = 0; j < NUM_STOCKS; ++j) {
            double weight = (i == j) ? 1.0 : true_correlations[i][j] * 0.2;
            total_shock += current_returns[j] * weight;
        }
        prices[i] *= (1.0 + total_shock);
        realized_returns[i] = total_shock;

        returns_history[i].push_back(total_shock);
        if ((int)returns_history[i].size() > HISTORY_LEN)
            returns_history[i].erase(returns_history[i].begin());
    }

    update_ewma(realized_returns);
    tick_count++;
}

struct LWResult {
    vector<vector<double>> cov;
    vector<vector<double>> corr;
    double shrinkage_intensity;
};

LWResult ledoit_wolf_shrink(const vector<vector<double>>& sample_cov, int T) {
    int p = sample_cov.size();
    LWResult result;
    result.cov.assign(p, vector<double>(p, 0.0));
    result.corr.assign(p, vector<double>(p, 0.0));

    double mu = 0.0;
    for (int i = 0; i < p; ++i) mu += sample_cov[i][i];
    mu /= p;

    double delta = 0.0;
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            double diff = sample_cov[i][j] - ((i == j) ? mu : 0.0);
            delta += diff * diff;
        }
    }
    delta /= (double)(p * p);

    int n_eff = max(T, 2);

    double beta = 0.0;
    if (!returns_history[0].empty()) {
        int n = (int)returns_history[0].size();
        vector<double> means(p, 0.0);
        for (int i = 0; i < p; ++i) {
            for (int t = 0; t < n; ++t) {
                means[i] += returns_history[i][t];
            }
            means[i] /= n;
        }

        double sum_sq = 0.0;
        for (int t = 0; t < n; ++t) {
            vector<double> z(p);
            for (int i = 0; i < p; ++i) z[i] = returns_history[i][t] - means[i];

            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < p; ++j) {
                    double outer = z[i] * z[j];
                    double diff = outer - sample_cov[i][j];
                    sum_sq += diff * diff;
                }
            }
        }
        beta = sum_sq / (n * n * p * p);
    }

    double intensity = (delta > 1e-15) ? min(beta / delta, 1.0) : 0.0;
    intensity = max(0.0, intensity);
    result.shrinkage_intensity = intensity;

    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            double target_val = (i == j) ? mu : 0.0;
            result.cov[i][j] = intensity * target_val + (1.0 - intensity) * sample_cov[i][j];
        }
    }

    for (int i = 0; i < p; ++i) {
        result.corr[i][i] = 1.0;
        double si = sqrt(max(result.cov[i][i], 1e-15));
        for (int j = i + 1; j < p; ++j) {
            double sj = sqrt(max(result.cov[j][j], 1e-15));
            double r = result.cov[i][j] / (si * sj);
            r = max(-1.0, min(1.0, r));
            result.corr[i][j] = result.corr[j][i] = r;
        }
    }

    return result;
}

LWResult estimate_correlations() {
    return ledoit_wolf_shrink(ewma_cov, tick_count);
}

double predicted_return(int i, const vector<vector<double>>& est) {
    if (returns_history[i].empty()) return 0.0;
    double weighted_sum = 0.0, weight_total = 0.0;
    for (int j = 0; j < NUM_STOCKS; ++j) {
        if (j == i || returns_history[j].empty()) continue;
        double w = fabs(est[i][j]);
        if (w < 1e-6) continue;
        double last_return_j = returns_history[j].back();
        weighted_sum += est[i][j] * last_return_j * w;
        weight_total += w;
    }
    if (weight_total < 1e-6) return 0.0;
    return weighted_sum / weight_total;
}

int main() {
    cout << "Starting Market Simulator..." << endl;
    init_data();
    
    httplib::Server svr;
    
    // Set static file mount point
    cout << "Setting up static file serving..." << endl;
    svr.set_mount_point("/", "./static");

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK", "text/plain");
    });

    svr.Get("/api/tick", [](const httplib::Request&, httplib::Response& res) {
        tick_simulation();
        res.set_content("OK", "text/plain");
    });

    svr.Get("/api/data", [](const httplib::Request&, httplib::Response& res) {
        auto lw = estimate_correlations();
        stringstream ss;
        ss << "[";
        for (int i = 0; i < NUM_STOCKS; ++i) {
            if (i > 0) ss << ",";
            double pred = predicted_return(i, lw.corr);
            double signal = tanh(pred / BASE_A);
            
