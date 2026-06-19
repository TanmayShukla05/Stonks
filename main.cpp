#include "httplib.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>

using namespace std;

// CONFIG
const int NUM_STOCKS = 12;
double BASE_A = 0.02; // 2% volatility 'a'
vector<double> prices(NUM_STOCKS, 100.0);
vector<string> modes(NUM_STOCKS, "normal"); // normal, crash, boost
vector<vector<double>> correlations(NUM_STOCKS, vector<double>(NUM_STOCKS, 0.0));
vector<vector<double>> returns_history; // For ML correlation guessing

// Simple RNG
mt19937 rng(1337);

void init_data() {
    // Fill diagonal with 1.0, random correlation elsewhere
    for(int i=0; i<NUM_STOCKS; ++i) {
        correlations[i][i] = 1.0;
        for(int j=i+1; j<NUM_STOCKS; ++j) {
            double c = (double)rand() / RAND_MAX * 0.5; // Random correlation
            correlations[i][j] = correlations[j][i] = c;
        }
        returns_history.push_back(vector<double>());
    }
}

// Bimodal Logic: Normal(peak, 0.005)
double get_bimodal_return(string mode) {
    double peak = (mode == "crash") ? -0.05 : (mode == "boost") ? 0.05 : 0.0;
    normal_distribution<double> dist(peak, 0.01);
    return dist(rng);
}

void tick_simulation() {
    vector<double> current_returns(NUM_STOCKS);
    for(int i=0; i<NUM_STOCKS; ++i) {
        current_returns[i] = get_bimodal_return(modes[i]);
        if(modes[i] != "normal") modes[i] = "normal"; // Reset mode after 1 tick
    }

    // Apply Correlated Impact
    for(int i=0; i<NUM_STOCKS; ++i) {
        for(int j=0; j<NUM_STOCKS; ++j) {
            prices[i] *= (1.0 + (current_returns[j] * correlations[i][j] * 0.2));
        }
        returns_history[i].push_back(current_returns[i]);
        if(returns_history[i].size() > 50) returns_history[i].erase(returns_history[i].begin());
    }
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
        stringstream ss;
        ss << "[";
        for(int i=0; i<NUM_STOCKS; ++i) {
            if (i > 0) ss << ",";
            ss << "{\"p\":" << prices[i] << "}";
        }
        ss << "]";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/api/action", [](const httplib::Request& req, httplib::Response& res) {
        // Simple parsing: ?id=0&type=boost
        auto id = stoi(req.get_param_value("id"));
        auto type = req.get_param_value("type");
        modes[id] = type;
        res.set_content("OK", "text/plain");
    });

    cout << "Server starting..." << endl;
    svr.listen("0.0.0.0", stoi(getenv("PORT") ? getenv("PORT") : "8080"));
}
