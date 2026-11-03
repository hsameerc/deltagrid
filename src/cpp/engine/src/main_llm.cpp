#include "deltagrid_llm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "   HSRU DELTAGRID EDGE: ZERO-KV CAUSAL LLM ENGINE   \n";
    std::cout << "====================================================\n\n";

    static const char* const kDefaultBins[] = {
        "models/deltagrid_weights.bin",
        "models/model_weights.bin",
        "models/weights.bin",
        "models/story_weights.bin"
    };
    const char* default_bin   = nullptr;
    const char* default_vocab = "models/vocab.txt";

    if (argc < 2) {
        for (const char* candidate : kDefaultBins) {
            std::ifstream fcheck(candidate, std::ios::binary);
            if (fcheck.good()) {
                default_bin = candidate;
                break;
            }
        }
        if (!default_bin) {
            std::cout << "Usage: " << argv[0] << " <llm_weights.bin> [vocab.txt] [gen_length]\n";
            return 1;
        }
    }

    const char* bin_path   = (argc > 1) ? argv[1] : default_bin;
    const char* vocab_path = (argc > 2) ? argv[2] : ((argc == 1) ? default_vocab : nullptr);
    int gen_length         = (argc > 3) ? std::atoi(argv[3]) : 100;

    std::cout << "[INFO] Loading DeltaGridLM from " << bin_path << "...\n";
    void* ctx = deltagrid_llm_init_context(bin_path);
    if (!ctx) {
        std::cerr << "[ERROR] Failed to initialize DeltaGridLM context from " << bin_path << "\n";
        return 1;
    }

    int vocab_size = deltagrid_llm_get_vocab_size(ctx);
    int d_model    = deltagrid_llm_get_d_model(ctx);
    int num_layers = deltagrid_llm_get_num_layers(ctx);
    std::cout << "[INFO] Model Loaded | Vocab: " << vocab_size
              << " | d_model: " << d_model
              << " | Layers: " << num_layers << "\n";

    std::vector<std::string> vocab(vocab_size);
    bool has_vocab = false;
    int bos_token = (vocab_size > 50256) ? 50256 : 0;
    int eos_token = (vocab_size > 50256) ? 50256 : -1;
    if (vocab_path != nullptr) {
        std::ifstream vfile(vocab_path);
        if (vfile.is_open()) {
            has_vocab = true;
            std::string line;
            int idx = 0;
            while (std::getline(vfile, line) && idx < vocab_size) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line == "<|endoftext|>" || line == "<|bos|>" || line == "<bos>" || line == "<s>") {
                    bos_token = idx;
                }
                if (line == "<|endoftext|>" || line == "<|eos|>" || line == "<eos>" || line == "</s>") {
                    eos_token = idx;
                }
                size_t pos = 0;
                while ((pos = line.find("\\n", pos)) != std::string::npos) {
                    line.replace(pos, 2, "\n");
                    pos += 1;
                }
                vocab[idx++] = line;
            }
        }
    }

    std::srand((unsigned)std::time(nullptr));
    int current_token = bos_token;

    std::vector<float> logits(vocab_size, 0.0f);
    std::vector<int> generated_tokens;
    generated_tokens.reserve(gen_length);

    struct TokenProb {
        float prob;
        int id;
    };
    std::vector<TokenProb> token_probs(vocab_size);

    std::cout << "\n[INFO] Generating " << gen_length << " tokens in strict O(1) RAM...\n\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    if (has_vocab && vocab_size > 7454 && vocab[7454] == "Once") {
        const int prompt_ids[] = {7454, 2402, 257, 640, 11, 612, 373, 257, 1310};
        for (int i = 0; i < 8; ++i) {
            std::cout << vocab[prompt_ids[i]] << std::flush;
            deltagrid_llm_step(ctx, prompt_ids[i], logits.data());
            generated_tokens.push_back(prompt_ids[i]);
        }
        current_token = prompt_ids[8];
        std::cout << vocab[current_token] << std::flush;
        generated_tokens.push_back(current_token);
    }

    for (int t = 0; t < gen_length; ++t) {
        deltagrid_llm_step(ctx, current_token, logits.data());

        // Repetition penalty (1.15)
        float rep_penalty = 1.15f;
        for (int past_id : generated_tokens) {
            if (logits[past_id] > 0.0f) {
                logits[past_id] /= rep_penalty;
            } else {
                logits[past_id] *= rep_penalty;
            }
        }

        // Top-K (40) & Temperature (0.8) sampling
        int top_k = std::min(40, vocab_size);
        float temperature = 0.8f;

        for (int v = 0; v < vocab_size; ++v) {
            token_probs[v].prob = logits[v] / temperature;
            token_probs[v].id   = v;
        }

        std::partial_sort(
            token_probs.begin(),
            token_probs.begin() + top_k,
            token_probs.end(),
            [](const TokenProb& a, const TokenProb& b) { return a.prob > b.prob; }
        );

        float max_logit = token_probs[0].prob;
        float sum_exp   = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            token_probs[i].prob = std::exp(token_probs[i].prob - max_logit);
            sum_exp += token_probs[i].prob;
        }
        for (int i = 0; i < top_k; ++i) {
            token_probs[i].prob /= sum_exp;
        }

        float r   = (float)std::rand() / (float)RAND_MAX;
        float cdf = 0.0f;
        int next_token = token_probs[0].id;
        for (int i = 0; i < top_k; ++i) {
            cdf += token_probs[i].prob;
            if (r <= cdf) {
                next_token = token_probs[i].id;
                break;
            }
        }

        current_token = next_token;
        generated_tokens.push_back(current_token);
        if (eos_token >= 0 && current_token == eos_token) {
            break;
        }

        if (has_vocab && !vocab[current_token].empty()) {
            std::cout << vocab[current_token] << std::flush;
        } else {
            std::cout << current_token << " " << std::flush;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n\n----------------------------------------------------\n";
    std::cout << "Generated " << gen_length << " tokens in " << elapsed.count() << " seconds.\n";
    std::cout << "Speed: " << (gen_length / std::max(elapsed.count(), 1e-9)) << " tokens/sec (Zero KV-Cache)\n";
    std::cout << "----------------------------------------------------\n";

    deltagrid_llm_free_context(ctx);
    return 0;
}

