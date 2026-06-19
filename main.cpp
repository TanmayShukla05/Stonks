#include "httplib.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>

using namespace std;

// ===================== CONFIG =====================
const int NUM_STOCKS = 12;
const double BASE_A = 0.02;
const int HISTORY_LEN = 50;
const int MCMC_ITERATIONS = 1000;
const int MCMC_BURNIN = 200;

vector<double> prices(NUM_STOCKS, 100.0);
vector<string> modes(NUM_STOCKS, "normal");
vector<vector<double>> true_correlations(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
vector<vector<double>> returns_history;

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

double get_bimodal_return(const string& mode) {
    double peakLow, peakHigh;
    if (mode == "crash") {
        peakLow = -6 * BASE_A;
        peakHigh = -4 * BASE_A;
    } else if (mode == "boost") {
        peakLow = 4 * BASE_A;
        peakHigh = 6 * BASE_A;
    } else {
        peakLow = -BASE_A;
        peakHigh = BASE_A;
    }

    uniform_real_distribution<double> coin(0.0, 1.0);
    double peak = (coin(rng) < 0.5) ? peakLow : peakHigh;
    normal_distribution<double> dist(peak, BASE_A * 0.25);
    return dist(rng);
}

void tick_simulation() {
    vector<double> current_returns(NUM_STOCKS);
    for (int i = 0; i < NUM_STOCKS; ++i) {
        current_returns[i] = get_bimodal_return(modes[i]);
        if (modes[i] != "normal") modes[i] = "normal";
    }

    for (int i = 0; i < NUM_STOCKS; ++i) {
        double total_shock = 0.0;
        for (int j = 0; j < NUM_STOCKS; ++j) {
            double weight = (i == j) ? 1.0 : true_correlations[i][j] * 0.2;
            total_shock += current_returns[j] * weight;
        }

        prices[i] *= (1.0 + total_shock);

        returns_history[i].push_back(total_shock);
        if ((int)returns_history[i].size() > HISTORY_LEN) {
            returns_history[i].erase(returns_history[i].begin());
        }
    }
}

double pearson(const vector<double>& a, const vector<double>& b) {
    int n = min(a.size(), b.size());
    if (n < 5) return 0.0;

    double ma = 0.0, mb = 0.0;
    for (int k = 0; k < n; ++k) {
        ma += a[k];
        mb += b[k];
    }
    ma /= n;
    mb /= n;

    double cov = 0.0, va = 0.0, vb = 0.0;
    for (int k = 0; k < n; ++k) {
        double da = a[k] - ma;
        double db = b[k] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }

    if (va < 1e-12 || vb < 1e-12) return 0.0;
    return cov / sqrt(va * vb);
}

vector<vector<double>> estimate_correlations_pearson() {
    vector<vector<double>> est(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
    for (int i = 0; i < NUM_STOCKS; ++i) {
        est[i][i] = 1.0;
        for (int j = i + 1; j < NUM_STOCKS; ++j) {
            double r = pearson(returns_history[i], returns_history[j]);
            est[i][j] = est[j][i] = r;
        }
    }
    return est;
}

double log_likelihood(const vector<vector<double>>& corr) {
    // Simple pseudo-likelihood based on agreement between observed co-movement
    // and candidate correlations.
    double ll = 0.0;
    for (int i = 0; i < NUM_STOCKS; ++i) {
        for (int j = i + 1; j < NUM_STOCKS; ++j) {
            double r = pearson(returns_history[i], returns_history[j]);
            double diff = corr[i][j] - r;
            ll -= diff * diff / 0.05;
        }
    }
    return ll;
}

vector<vector<double>> estimate_correlations_mcmc() {
    vector<vector<double>> current = estimate_correlations_pearson();
    double current_ll = log_likelihood(current);

    vector<vector<double>> sum_corr(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
    int samples = 0;

    uniform_real_distribution<double> uni(0.0, 1.0);
    normal_distribution<double> proposal(0.0, 0.05);

    for (int iter = 0; iter < MCMC_ITERATIONS; ++iter) {
        vector<vector<double>> proposed = current;

        int i = (int)(uni(rng) * NUM_STOCKS);
        int j = (int)(uni(rng) * NUM_STOCKS);
        if (i == j) j = (j + 1) % NUM_STOCKS;
        if (i > j) swap(i, j);

        double nv = proposed[i][j] + proposal(rng);
        nv = max(-0.99, min(0.99, nv));
        proposed[i][j] = proposed[j][i] = nv;

        double prop_ll = log_likelihood(proposed);
        double log_alpha = prop_ll - current_ll;

        if (log(uni(rng)) < log_alpha) {
            current = proposed;
            current_ll = prop_ll;
        }

        if (iter >= MCMC_BURNIN) {
            for (int a = 0; a < NUM_STOCKS; ++a) {
                for (int b = 0; b < NUM_STOCKS; ++b) {
                    sum_corr[a][b] += current[a][b];
                }
            }
            samples++;
        }
    }

    if (samples > 0) {
        for (int a = 0; a < NUM_STOCKS; ++a) {
            for (int b = 0; b < NUM_STOCKS; ++b) {
                sum_corr[a][b] /= samples;
            }
        }
    }

    for (int i = 0; i < NUM_STOCKS; ++i) sum_corr[i][i] = 1.0;
    return sum_corr;
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
            double signal = tanh(pred / BASE_A);
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
                if (i != j) {
                    total_abs_error += fabs(err);
                    pair_count++;
                }
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

    int port = getenv("PORT") ? stoi(getenv("PORT")) : 8080;
    cout << "Server starting on port " << port << endl;
    svr.listen("0.0.0.0", port);
}
