#include "../../hnswlib/hnswlib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <mutex>
#include <nlohmann/json.hpp>

namespace {



struct Config {
    std::string dataset = "glove-200-angular";
    std::string data_dir = "../../data";
    std::string metric = "";  // empty: infer from dataset
    std::string index_path = "";
    std::string python_bin = "";

    size_t M = 16;
    size_t ef_construction = 200;
    size_t top_k = 10;
    std::vector<size_t> efs{70, 90, 100, 120, 140};

    size_t train_limit = 0;  // 0: all
    size_t test_limit = 0;   // 0: all
    int num_threads = -1;    // <=0 means use hardware concurrency
    bool build_only = false;
    bool use_adaptive_ef = false;
    size_t topk_clusters=20;
};

struct DataSet {
    std::vector<float> train;
    std::vector<float> test;
    std::vector<hnswlib::labeltype> neighbors;
    size_t train_rows = 0;
    size_t test_rows = 0;
    size_t dim = 0;
    size_t neighbors_k = 0;
};

    nlohmann::json topcands_to_json(const std::vector<std::vector<int>>& top_cands) {
        nlohmann::json rows = nlohmann::json::array();
        for (const auto& cand_row : top_cands) {
            nlohmann::json row = nlohmann::json::array();
            for (int node_id : cand_row) {
                row.push_back(node_id);
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

    nlohmann::json label_set_to_json(const std::vector<std::unordered_set<hnswlib::labeltype>>& labels) {
        nlohmann::json rows = nlohmann::json::array();
        for (const auto& label_set : labels) {
            std::vector<hnswlib::labeltype> sorted(label_set.begin(), label_set.end());
            std::sort(sorted.begin(), sorted.end());

            nlohmann::json row = nlohmann::json::array();
            for (hnswlib::labeltype label : sorted) {
                row.push_back(label);
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

    void save_logged_efs_json(const std::string& path,
                      const std::vector<int>& logged_efs,
                      const std::vector<std::vector<int>>& top_cands,
                      const std::vector<std::unordered_set<hnswlib::labeltype>>& preds,
                      const std::vector<std::unordered_set<hnswlib::labeltype>>& gts,
                      const Config& cfg,
                      size_t ef) {
        nlohmann::json j;
        j["dataset"] = cfg.dataset;
        j["metric"] = cfg.metric;
        j["top_k"] = cfg.top_k;
        j["ef"] = ef;
        j["use_adaptive_ef"] = cfg.use_adaptive_ef;
        j["topk_clusters"] = cfg.topk_clusters;
        j["logged_efs"] = logged_efs;
        j["top_cands"] = topcands_to_json(top_cands);
        j["pred"] = label_set_to_json(preds);
        j["gt"] = label_set_to_json(gts);
        j["n"] = logged_efs.size();

        std::ofstream out(path);
        if (!out) throw std::runtime_error("failed to open for write: " + path);
        out << j.dump(2) << "\n";                     // pretty print
    }

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t num_threads, Function fn) {
    if (num_threads <= 0) num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    if (num_threads == 1) {
        for (size_t id = start; id < end; ++id) fn(id, 0);
        return;
    }

    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);
    std::exception_ptr last_exception = nullptr;
    std::mutex last_exception_mu;

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            while (true) {
                size_t id = current.fetch_add(1);
                if (id >= end) break;
                try {
                    fn(id, thread_id);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(last_exception_mu);
                    last_exception = std::current_exception();
                    current = end;
                    break;
                }
            }
        });
    }

    for (auto& t : threads) t.join();
    if (last_exception) std::rethrow_exception(last_exception);
}

std::vector<size_t> parse_efs(const std::string& text) {
    std::vector<size_t> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        out.push_back(static_cast<size_t>(std::stoul(token)));
    }
    if (out.empty()) throw std::runtime_error("--efs requires at least one value");
    return out;
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto need_value = [&](const std::string& key) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return std::string(argv[++i]);
        };

        if (arg == "--dataset") cfg.dataset = need_value(arg);
        else if (arg == "--data_dir") cfg.data_dir = need_value(arg);
        else if (arg == "--metric") cfg.metric = need_value(arg);
        else if (arg == "--index_path") cfg.index_path = need_value(arg);
        else if (arg == "--python_bin") cfg.python_bin = need_value(arg);
        else if (arg == "--M") cfg.M = static_cast<size_t>(std::stoul(need_value(arg)));
        else if (arg == "--efc") cfg.ef_construction = static_cast<size_t>(std::stoul(need_value(arg)));
        else if (arg == "--top_k") cfg.top_k = static_cast<size_t>(std::stoul(need_value(arg)));
        else if (arg == "--efs") cfg.efs = parse_efs(need_value(arg));
        else if (arg == "--train_limit") cfg.train_limit = static_cast<size_t>(std::stoull(need_value(arg)));
        else if (arg == "--test_limit") cfg.test_limit = static_cast<size_t>(std::stoull(need_value(arg)));
        else if (arg == "--num_threads") cfg.num_threads = std::stoi(need_value(arg));
        else if (arg == "--build_only") cfg.build_only = true;
        else if (arg == "--adaptive_ef") cfg.use_adaptive_ef = true;
        else if (arg == "--topk_clusters") cfg.topk_clusters = static_cast<size_t>(std::stoull(need_value(arg)));
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./example_hnsw_test [options]\n"
                << "  --dataset <name>      dataset id (default: glove-50-angular)\n"
                << "  --data_dir <path>     hdf5 directory (default: ../../data)\n"
                << "  --metric <l2|cosine>  override metric (default: infer from dataset)\n"
                << "  --index_path <path>   load if exists, else build and save\n"
                << "  --python_bin <path>   python executable with h5py installed\n"
                << "  --M <int>             M (default: 16)\n"
                << "  --efc <int>           efConstruction (default: 200)\n"
                << "  --top_k <int>         top-k (default: 10)\n"
                << "  --efs <a,b,c>         efSearch list (default: 100,800)\n"
                << "  --train_limit <int>   use first N train vectors (0: all)\n"
                << "  --test_limit <int>    use first N test vectors (0: all)\n"
                << "  --num_threads <int>   build threads (default: hardware concurrency)\n"
                << "  --build_only          build/load index only\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (cfg.top_k == 0) throw std::runtime_error("--top_k must be > 0");
    return cfg;
}

std::string infer_metric(const std::string& dataset_name) {
    if (dataset_name.find("euclidean") != std::string::npos) return "l2";
    return "cosine";
}

std::string build_hdf5_path(const Config& cfg) {
    return cfg.data_dir + "/" + cfg.dataset + ".hdf5";
}

std::string sh_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

bool can_import_h5py(const std::string& python_bin) {
    if (python_bin.empty()) return false;
    std::ostringstream cmd;
    cmd << sh_quote(python_bin) << " -c " << sh_quote("import h5py") << " >/dev/null 2>&1";
    return std::system(cmd.str().c_str()) == 0;
}

std::string resolve_python_with_h5py(const Config& cfg) {
    std::vector<std::string> candidates;

    auto push_unique = [&](const std::string& s) {
        if (s.empty()) return;
        for (const auto& x : candidates) {
            if (x == s) return;
        }
        candidates.push_back(s);
    };

    push_unique(cfg.python_bin);

    const char* env_python = std::getenv("HNSW_H5PY_PYTHON");
    if (env_python) push_unique(std::string(env_python));

    const char* conda_prefix = std::getenv("CONDA_PREFIX");
    if (conda_prefix) push_unique(std::string(conda_prefix) + "/bin/python");

    const char* venv_prefix = std::getenv("VIRTUAL_ENV");
    if (venv_prefix) push_unique(std::string(venv_prefix) + "/bin/python");

    push_unique("python");
    push_unique("python3");

    for (const auto& py : candidates) {
        if (can_import_h5py(py)) return py;
    }

    std::ostringstream oss;
    oss << "no python interpreter with h5py found. Tried:";
    for (const auto& py : candidates) oss << " " << py;
    oss << ". Use --python_bin <path> (example: /home/dongseob/miniconda3/envs/vdb/bin/python)";
    throw std::runtime_error(oss.str());
}

void write_text_file(const std::string& path, const std::string& content) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) throw std::runtime_error("failed to write file: " + path);
    out << content;
}

std::unordered_map<std::string, size_t> read_meta_kv(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("failed to open meta file: " + path);

    std::unordered_map<std::string, size_t> kv;
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = line.substr(0, pos);
        const std::string value = line.substr(pos + 1);
        kv[key] = static_cast<size_t>(std::stoull(value));
    }
    return kv;
}

template <typename T>
std::vector<T> read_binary_vec(const std::string& path, size_t expected_count) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error("failed to open binary file: " + path);

    std::vector<T> v(expected_count);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (!in) throw std::runtime_error("failed to read binary file: " + path);
    return v;
}

DataSet load_dataset_via_python_h5py(const Config& cfg) {
    const std::string h5_path = build_hdf5_path(cfg);
    const std::string python_bin = resolve_python_with_h5py(cfg);

    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const std::string uniq = std::to_string(static_cast<unsigned long long>(::getpid())) + "_" + std::to_string(tid);
    const std::string tmp_dir = "./tmp/hnsw_h5dump_" + uniq;
    const std::string script_path = "./tmp/hnsw_h5dump_" + uniq + ".py";

    const std::string py = R"PY(
import os
import sys
import numpy as np
import h5py

h5_path = sys.argv[1]
out_dir = sys.argv[2]
train_limit = int(sys.argv[3])
test_limit = int(sys.argv[4])

os.makedirs(out_dir, exist_ok=True)

with h5py.File(h5_path, "r") as f:
    train = f["train"]
    test = f["test"]
    neighbors = f["neighbors"]

    train_rows = train.shape[0] if train_limit <= 0 else min(train.shape[0], train_limit)
    test_rows = test.shape[0] if test_limit <= 0 else min(test.shape[0], test_limit)
    dim = int(train.shape[1])
    if int(test.shape[1]) != dim:
        raise RuntimeError("train/test dimension mismatch")

    train_arr = np.asarray(train[:train_rows], dtype=np.float32, order="C")
    test_arr = np.asarray(test[:test_rows], dtype=np.float32, order="C")
    neigh_arr = np.asarray(neighbors[:test_rows], dtype=np.uint32, order="C")

    train_arr.tofile(os.path.join(out_dir, "train.bin"))
    test_arr.tofile(os.path.join(out_dir, "test.bin"))
    neigh_arr.tofile(os.path.join(out_dir, "neighbors.bin"))

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as meta:
        meta.write(f"train_rows={train_rows}\n")
        meta.write(f"test_rows={test_rows}\n")
        meta.write(f"dim={dim}\n")
        meta.write(f"neighbors_k={int(neighbors.shape[1])}\n")
)PY";

    write_text_file(script_path, py);

    std::ostringstream cmd;
    cmd << sh_quote(python_bin) << " "
        << sh_quote(script_path) << " "
        << sh_quote(h5_path) << " "
        << sh_quote(tmp_dir) << " "
        << cfg.train_limit << " "
        << cfg.test_limit;

    std::cout << "loading data via h5py: " << h5_path << "\n";
    std::cout << "python for h5py: " << python_bin << "\n";
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::remove(script_path.c_str());
        throw std::runtime_error("h5py loader script failed with code " + std::to_string(rc));
    }

    DataSet ds;
    const auto meta = read_meta_kv(tmp_dir + "/meta.txt");
    ds.train_rows = meta.at("train_rows");
    ds.test_rows = meta.at("test_rows");
    ds.dim = meta.at("dim");
    ds.neighbors_k = meta.at("neighbors_k");

    ds.train = read_binary_vec<float>(tmp_dir + "/train.bin", ds.train_rows * ds.dim);
    ds.test = read_binary_vec<float>(tmp_dir + "/test.bin", ds.test_rows * ds.dim);
    const auto neighbors_u32 = read_binary_vec<uint32_t>(tmp_dir + "/neighbors.bin", ds.test_rows * ds.neighbors_k);
    ds.neighbors.resize(neighbors_u32.size());
    for (size_t i = 0; i < neighbors_u32.size(); ++i) {
        ds.neighbors[i] = static_cast<hnswlib::labeltype>(neighbors_u32[i]);
    }

    std::remove(script_path.c_str());

    std::cout << "data loaded: train=" << ds.train_rows
              << ", test=" << ds.test_rows
              << ", dim=" << ds.dim
              << ", neighbors_k=" << ds.neighbors_k << "\n";

    return ds;
}

std::unordered_set<hnswlib::labeltype>
make_gt_set(const std::vector<hnswlib::labeltype>& neighbors, size_t row, size_t stride, size_t top_k,
            size_t train_rows) {
    std::unordered_set<hnswlib::labeltype> gt;
    const size_t base = row * stride;
    for (size_t i = 0; i < top_k; ++i) {
        const hnswlib::labeltype id = neighbors[base + i];
        if (static_cast<size_t>(id) < train_rows) gt.insert(id);
    }
    return gt;
}

std::unordered_set<hnswlib::labeltype>
make_pred_set(std::priority_queue<std::pair<float, hnswlib::labeltype>> pq) {
    std::unordered_set<hnswlib::labeltype> pred;
    while (!pq.empty()) {
        pred.insert(pq.top().second);
        pq.pop();
    }
    return pred;
}

size_t count_intersection(const std::unordered_set<hnswlib::labeltype>& a,
                          const std::unordered_set<hnswlib::labeltype>& b) {
    const auto& small = (a.size() < b.size()) ? a : b;
    const auto& large = (a.size() < b.size()) ? b : a;
    size_t cnt = 0;
    for (auto x : small) {
        if (large.find(x) != large.end()) cnt++;
    }
    return cnt;
}

size_t estimate_layer0_undirected_edges(const hnswlib::HierarchicalNSW<float>& index) {
    size_t directed = 0;
    for (size_t i = 0; i < index.cur_element_count; ++i) {
        auto* ll = index.get_linklist0(static_cast<hnswlib::tableint>(i));
        directed += index.getListCount(ll);
    }
    return directed / 2;
}

void normalize_rows(std::vector<float>* data, size_t rows, size_t dim) {
    for (size_t r = 0; r < rows; ++r) {
        float* row = data->data() + r * dim;
        double sq = 0.0;
        for (size_t d = 0; d < dim; ++d) sq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        const double norm = std::sqrt(sq);
        if (norm <= 0.0) continue;
        const float inv = static_cast<float>(1.0 / norm);
        for (size_t d = 0; d < dim; ++d) row[d] *= inv;
    }
}

double elapsed_seconds(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void print_progress(const std::string& stage, size_t current, size_t total) {
    if (total == 0) return;
    const double pct = 100.0 * static_cast<double>(current) / static_cast<double>(total);
    std::cout << "\r[" << stage << "] "
              << current << "/" << total
              << " (" << std::fixed << std::setprecision(1) << pct << "%)"
              << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto t_all = std::chrono::steady_clock::now();
        Config cfg = parse_args(argc, argv);
        if (cfg.metric.empty()) cfg.metric = infer_metric(cfg.dataset);

        DataSet ds = load_dataset_via_python_h5py(cfg);
        if (cfg.metric == "cosine") {
            normalize_rows(&ds.train, ds.train_rows, ds.dim);
            normalize_rows(&ds.test, ds.test_rows, ds.dim);
        }

        if (cfg.top_k > ds.neighbors_k) {
            throw std::runtime_error("--top_k exceeds neighbors columns in dataset");
        }

        std::unique_ptr<hnswlib::SpaceInterface<float>> space;
        if (cfg.metric == "l2") {
            space.reset(new hnswlib::L2Space(ds.dim));
        } else if (cfg.metric == "cosine") {
            space.reset(new hnswlib::InnerProductSpace(ds.dim));
        } else {
            throw std::runtime_error("--metric must be one of: l2, cosine");
        }

        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        const bool load_index = !cfg.index_path.empty();
        if (load_index) {
            FILE* f = std::fopen(cfg.index_path.c_str(), "rb");
            if (f) {
                std::fclose(f);
                std::cout << "loading index: " << cfg.index_path << "\n";
                index.reset(new hnswlib::HierarchicalNSW<float>(space.get(), cfg.index_path));
            }
        }

        if (!index) {
            std::cout << "building index: train_size=" << ds.train_rows << "\n";
            const auto t_build = std::chrono::steady_clock::now();
            index.reset(new hnswlib::HierarchicalNSW<float>(
                space.get(), ds.train_rows, cfg.M, cfg.ef_construction));
            const size_t build_step = std::max<size_t>(1, ds.train_rows / 100);
            size_t build_threads = (cfg.num_threads > 0)
                ? static_cast<size_t>(cfg.num_threads)
                : std::thread::hardware_concurrency();
            if (build_threads == 0) build_threads = 1;
            if (ds.train_rows <= build_threads * 4) build_threads = 1;
            std::cout << "build threads: " << build_threads << "\n";

            std::atomic<size_t> built(0);
            std::mutex progress_mu;

            size_t start = 0;
            if (ds.train_rows > 0) {
                index->addPoint(ds.train.data(), static_cast<hnswlib::labeltype>(0));
                start = 1;
                built.store(1);
                print_progress("build", 1, ds.train_rows);
            }

            ParallelFor(start, ds.train_rows, build_threads, [&](size_t i, size_t /*thread_id*/) {
                index->addPoint(ds.train.data() + i * ds.dim, static_cast<hnswlib::labeltype>(i));
                const size_t done = built.fetch_add(1) + 1;
                if (done % build_step == 0 || done == ds.train_rows) {
                    std::lock_guard<std::mutex> lk(progress_mu);
                    print_progress("build", done, ds.train_rows);
                }
            });
            std::cout << "\n";
            std::cout << "build time: " << std::fixed << std::setprecision(3)
                      << elapsed_seconds(t_build) << " s\n";
            if (!cfg.index_path.empty()) {
                index->saveIndex(cfg.index_path);
                std::cout << "saved index: " << cfg.index_path << "\n";
            }
        }

        std::cout << "index info: elements=" << index->cur_element_count
                  << ", layer0_edges~=" << estimate_layer0_undirected_edges(*index) << "\n";

        if (cfg.build_only) {
            std::cout << "build_only=true, done\n";
            return 0;
        }

        // load cluster info
        std::map<int, std::vector<int>> query_to_cluster;

        const std::string path = cfg.data_dir + "/cluster_" + cfg.dataset + ".json";
        std::ifstream in(path);
        nlohmann::json root;
        in >> root;  // { "index2cluster": {...}, "query2cluster": {...} }

        // index2cluster: { "123": 7, ... } -> map<int,int>
        const auto& i2c = root.at("index2cluster");
        for (auto it = i2c.begin(); it != i2c.end(); ++it) {
            index->setNodeCluster(static_cast<hnswlib::tableint>(std::stoi(it.key())), it.value().get<int>());
        }
        index->refreshNodeClusterDenseCache();

        // query2cluster: { "0": [1,2,3], ... } -> map<int, vector<int>>
        const auto& q2c = root.at("query2cluster");
        if (cfg.use_adaptive_ef) {
            for (auto it = q2c.begin(); it != q2c.end(); ++it) {
                auto v = it.value().get<std::vector<int>>();
                const size_t K = std::min<size_t>(cfg.topk_clusters, v.size());
                query_to_cluster[std::stoi(it.key())] =
                    std::vector<int>(v.begin(), v.begin() + K);
            }
        }


        for (size_t ef : cfg.efs) {
            const auto t_ef = std::chrono::steady_clock::now();
            index->setEf(std::max(ef, cfg.top_k));
            index->logged_efs = std::vector<int>();
            index->logged_topcands = std::vector<std::vector<int>>();
            double recall_sum = 0.0;
            double visited_sum = 0.0;
            double ms_sum = 0.0;

            std::cout << "testing ef=" << ef << ", use_adaptive_ef="<<cfg.use_adaptive_ef<<"\n";
            auto preds = std::vector<std::unordered_set<hnswlib::labeltype>>(ds.test_rows);
            auto gts = std::vector<std::unordered_set<hnswlib::labeltype>>(ds.test_rows);
            const size_t test_step = std::max<size_t>(1, ds.test_rows / 100);
            for (size_t i = 0; i < ds.test_rows; ++i) {
                const float* query = ds.test.data() + i * ds.dim;

                const long before = index->metric_distance_computations.load();
                const auto t0 = std::chrono::steady_clock::now();

                auto result = index->searchKnn(query, cfg.top_k, nullptr, cfg.use_adaptive_ef, query_to_cluster[i]);
                const auto t1 = std::chrono::steady_clock::now();
                const long after = index->metric_distance_computations.load();

                auto pred = make_pred_set(std::move(result));
                auto gt = make_gt_set(ds.neighbors, i, ds.neighbors_k, cfg.top_k, ds.train_rows);
                preds[i] = pred;
                gts[i] = gt;

                const size_t hit = count_intersection(pred, gt);
                recall_sum += static_cast<double>(hit) / static_cast<double>(cfg.top_k);
                visited_sum += static_cast<double>(after - before);
                ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();

                const size_t done = i + 1;
                if (done % test_step == 0 || done == ds.test_rows) {
                    print_progress("test", done, ds.test_rows);
                }

            }
            std::cout << "\n";

            const double avg_recall = recall_sum / static_cast<double>(ds.test_rows);
            const double avg_visited = visited_sum / static_cast<double>(ds.test_rows);
            const double avg_ms = ms_sum / static_cast<double>(ds.test_rows);


            std::vector<int> filtered;
            filtered.reserve(index->logged_efs.size());

            for (int v : index->logged_efs) {
                if (v != ef) filtered.push_back(v);
            }

            // 필요하면 메모리 줄이기
            filtered.shrink_to_fit();

            std::cout << "ef=" << ef
                      << ", recall=" << std::fixed << std::setprecision(4) << avg_recall
                      << ", visited=" << std::fixed << std::setprecision(4) << avg_visited
                      << ", avg_time_ms=" << std::fixed << std::setprecision(4) << avg_ms
                      << ", test_time_s=" << std::fixed << std::setprecision(4) << elapsed_seconds(t_ef)
                        << ", cnt_adapted=" << filtered.size()
                        << ", avg_adapted_ef="
                        << ([&]{
                                const auto& v = filtered;
                                if (v.empty()) return 0.0;
                                double s = std::accumulate(v.begin(), v.end(), 0.0);
                                return s / static_cast<double>(v.size());
                            }())
            << ", max_adapted_ef=" << ([&]{
                                const auto& v = filtered;
                                if (v.empty()) return 0;
                                return *std::max_element(v.begin(), v.end());
                            }())
            << ", min_adapted_ef=" << ([&]{
                                const auto& v = filtered;
                                if (v.empty()) return 0;
                                return *std::min_element(v.begin(), v.end());
                            }())
                      << "\n";

            std::string out_path = cfg.data_dir + "/logged_efs_" + cfg.dataset
                        + "_ef" + std::to_string(ef)
                        + (cfg.use_adaptive_ef ? "_adaptive" : "_vanilla")
                        + ".json";

            save_logged_efs_json(out_path, index->logged_efs, index->logged_topcands, preds, gts, cfg, ef);
            std::cout << "saved logged_efs json: " << out_path << "\n";
        }

        std::cout << "total time: " << std::fixed << std::setprecision(4)
                  << elapsed_seconds(t_all) << " s\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
