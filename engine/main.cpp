#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include "common.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <map>
#include <limits>
#include <optional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

using json = nlohmann::json;
static constexpr int protocol = 1;
static constexpr size_t max_line = 64 * 1024 * 1024;

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct Inbox {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<json> commands;
    bool eof = false;

    void read() {
        std::string line;
        while (std::getline(std::cin, line)) {
            json command;
            try {
                require(line.size() <= max_line, "command exceeds 64 MiB limit");
                command = json::parse(line);
                require(command.is_object(), "command must be an object");
            } catch (const std::exception & e) {
                command = {{"id", "invalid"}, {"op", "invalid"}, {"parse_error", e.what()}};
            }
            std::unique_lock lock(mutex);
            // Bounded private-pipe queue: backpressure, not unbounded allocation.
            ready.wait(lock, [&] { return commands.size() < 128; });
            commands.push_back(std::move(command));
            lock.unlock();
            ready.notify_all();
        }
        { std::lock_guard lock(mutex); eof = true; }
        ready.notify_all();
    }

    json next() {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return !commands.empty() || eof; });
        if (commands.empty()) return nullptr;
        json item = std::move(commands.front());
        commands.pop_front();
        lock.unlock();
        ready.notify_all();
        return item;
    }

    std::deque<json> drain(bool & closed) {
        std::deque<json> result;
        { std::lock_guard lock(mutex); result.swap(commands); closed = eof; }
        ready.notify_all();
        return result;
    }
};

struct Capture {
    bool active = false;
    int width = 0;
    std::set<int> layers;
    std::map<int, std::vector<float>> values;
    std::string error;

    static bool callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
        auto & self = *static_cast<Capture *>(data);
        if (!self.active) return ask ? false : true;
        int layer = -1;
        for (int candidate : self.layers) {
            if (std::string(tensor->name) == "l_out-" + std::to_string(candidate)) {
                layer = candidate;
                break;
            }
        }
        if (ask) return layer >= 0;
        if (layer < 0) return true;
        try {
            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor),
                    "capture tensor is not contiguous F32");
            require(tensor->ne[0] == self.width && tensor->ne[1] == 1 &&
                    tensor->ne[2] == 1 && tensor->ne[3] == 1,
                    "capture tensor has wrong shape: expected [n_embd,1,1,1]");
            require(!self.values.count(layer), "duplicate capture for layer " + std::to_string(layer));
            std::vector<float> values(self.width);
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            for (float value : values) require(std::isfinite(value), "non-finite activation");
            self.values.emplace(layer, std::move(values));
        } catch (const std::exception & e) {
            self.error = e.what();
            return false;
        } catch (...) {
            self.error = "capture allocation failed";
            return false;
        }
        return true;
    }
};

class Engine {
    Inbox & inbox;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    common_chat_templates_ptr templates;
    Capture capture;
    int width = 0;
    int layers = 0;
    int batch_size = 512;
    std::string active_id;
    std::string model_path;
    int64_t revision = 0;
    bool cancelled = false;
    std::string pending_utf8;
    std::string awaiting_tool;
    json tool_reply = nullptr;

    void emit(const std::string & id, const std::string & event, json data = json::object()) {
        data["v"] = protocol;
        data["id"] = id;
        data["event"] = event;
        std::cout << data.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
    }

    void ensure_loaded() { require(model && context, "no model loaded"); }

    json runtime() const {
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        return {{"engine_revision", TORMENT_ENGINE_REV}, {"wrapper_sha256", TORMENT_WRAPPER_SHA256}, {"protocol", protocol},
                {"context", context ? llama_n_ctx(context) : 0}, {"batch", batch_size},
                {"microbatch", context ? llama_n_ubatch(context) : 0},
                {"n_layer", layers}, {"n_embd", width}, {"cache_k", "F16"}, {"cache_v", "F16"},
                {"system", llama_print_system_info()}, {"peak_rss_bytes", usage.ru_maxrss},
                {"build", std::string(__DATE__) + " " + __TIME__}};
    }

    void reset() {
        llama_synchronize(context);
        require(llama_set_adapter_cvec(context, nullptr, 0, width, -1, -1) == 0,
                "failed to disable steering");
        llama_memory_clear(llama_get_memory(context), true);
        capture.active = false;
        capture.values.clear();
        capture.error.clear();
        revision = 0;
        cancelled = false;
    }

    std::vector<float> make_buffer(const json & rows) {
        require(rows.is_array(), "control rows must be an array");
        std::vector<float> buffer(static_cast<size_t>(layers - 1) * width, 0.0f);
        std::set<int> seen;
        for (const auto & row : rows) {
            int layer = row.at("layer").get<int>();
            require(layer >= 1 && layer < layers, "unsupported steering layer (layer 0 is unavailable)");
            require(seen.insert(layer).second, "duplicate control layer");
            const auto & values = row.at("values");
            require(values.is_array() && values.size() == static_cast<size_t>(width), "wrong control width");
            for (int i = 0; i < width; ++i) {
                float value = values.at(i).get<float>();
                require(std::isfinite(value), "non-finite control value");
                // Prism's legacy control buffer omits graph layer 0.
                buffer[static_cast<size_t>(layer - 1) * width + i] = value;
            }
        }
        return buffer;
    }

    void apply(const json & controls, size_t first_token, bool announce = true) {
        auto buffer = make_buffer(controls.value("rows", json::array()));
        int64_t next_revision = controls.value("revision", int64_t(0));
        require(next_revision >= 0, "negative control revision");
        llama_synchronize(context);
        bool any = std::any_of(buffer.begin(), buffer.end(), [](float v) { return v != 0.0f; });
        // Always write every supported row, including zeros. Clearing the adapter alone
        // leaves old device rows allocated and cannot stand in for replacing the buffer.
        require(llama_set_adapter_cvec(context, buffer.data(), buffer.size(), width, 1, layers - 1) == 0,
                "control-vector allocation or upload failed");
        if (!any) {
            require(llama_set_adapter_cvec(context, nullptr, 0, width, -1, -1) == 0,
                    "failed to disable all-zero controls");
        }
        revision = next_revision;
        if (announce) emit(active_id, "applied", {{"revision", revision}, {"first_token_index", first_token}});
    }

    void boundary(size_t first_token, bool allow_controls) {
        bool closed = false;
        auto commands = inbox.drain(closed);
        cancelled = cancelled || closed;
        json latest = nullptr;
        for (const auto & command : commands) {
            const std::string id = command.value("id", "invalid");
            try {
                require(command.value("v", 0) == protocol, "unsupported protocol version");
                const std::string op = command.value("op", "");
                require(command.value("target", "") == active_id, "engine busy; target active command explicitly");
                if (op == "cancel") {
                    cancelled = true;
                    emit(id, "result", {{"cancel_requested", true}});
                } else if (op == "tool_result" && !awaiting_tool.empty()) {
                    require(command.value("tool_id", "") == awaiting_tool, "tool call identity changed");
                    require(tool_reply.is_null(), "tool already answered");
                    tool_reply = command.at("result");
                    emit(id, "result", {{"received", true}});
                } else if (op == "controls" && allow_controls) {
                    int64_t next = command.at("revision").get<int64_t>();
                    int64_t prior = latest.is_null() ? revision : latest.at("revision").get<int64_t>();
                    require(next > prior, "control revisions must increase");
                    (void)make_buffer(command.at("rows")); // Validate every snapshot even if coalesced.
                    if (!latest.is_null()) emit(latest.at("id"), "result", {{"coalesced", true}});
                    latest = command;
                } else {
                    throw std::runtime_error("engine busy; only cancellation and live controls are allowed");
                }
            } catch (const std::exception & e) { emit(id, "error", {{"error", e.what()}}); }
        }
        if (!latest.is_null() && !cancelled) {
            try {
                apply(latest, first_token);
                emit(latest.at("id"), "result", {{"revision", revision}, {"first_token_index", first_token}});
            } catch (const std::exception & e) { emit(latest.at("id"), "error", {{"error", e.what()}}); }
        } else if (!latest.is_null()) {
            emit(latest.at("id"), "error", {{"error", "generation cancelled before control application"}});
        }
    }

    std::string render(const json & command) {
        const auto & messages = command.at("messages");
        require(messages.is_array() && !messages.empty() && messages.size() <= 1024, "invalid messages");
        for (const auto & message : messages) {
            require(message.is_object() && message.contains("role") && message.at("role").is_string() &&
                    message.contains("content") && message.at("content").is_string(), "message requires text role and content");
            const auto role = message.at("role").get<std::string>();
            require(role == "system" || role == "user" || role == "assistant", "unsupported message role");
        }
        if (command.value("raw", false)) {
            // Preserve the explicitly supplied completion prefix byte-for-byte.
            if (messages.size() == 1 && messages.at(0).at("role") == "user") {
                return messages.at(0).at("content").get<std::string>();
            }
            // Template-free conversation/factory contexts use a deliberately
            // plain, deterministic format, never an implicit model template.
            std::string prompt;
            for (const auto & message : messages) {
                prompt += message.at("role").get<std::string>() + ": " +
                          message.at("content").get<std::string>() + "\n\n";
            }
            return prompt + "assistant: ";
        }
        require(templates != nullptr, "model has no chat template; explicitly select raw completion mode");
        common_chat_templates_inputs inputs;
        inputs.enable_thinking = false;
        inputs.add_generation_prompt = true;
        inputs.now = std::chrono::system_clock::time_point{}; // deterministic render, recorded below
        for (const auto & message : messages) {
            common_chat_msg msg;
            msg.role = message.at("role").get<std::string>();
            msg.content = message.at("content").get<std::string>();
            inputs.messages.push_back(std::move(msg));
        }
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }

    std::vector<llama_token> tokenize(const std::string & text, bool allow_long = false) {
        require(text.size() <= 8 * 1024 * 1024, "prompt exceeds 8 MiB limit");
        const auto * vocab = llama_model_get_vocab(model);
        // Do not implicitly append EOS: the last token must really be assistant content.
        auto tokens = common_tokenize(vocab, text, false, true);
        if (llama_vocab_get_add_bos(vocab) && (tokens.empty() || tokens.front() != llama_vocab_bos(vocab))) {
            tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
        }
        require(!tokens.empty(), "prompt tokenized to no tokens");
        require(allow_long || tokens.size() < llama_n_ctx(context), "prompt exceeds context; reduce input or increase context");
        return tokens;
    }

    // Derive only role boundaries from the model's template. Tokenize the actual
    // result separately, with special-token parsing disabled: result data cannot
    // manufacture chat delimiters. Existing recurrent/KV state is not replayed.
    std::pair<std::string, std::string> tool_boundaries(const json & history, size_t call_index) {
        if (history.value("raw", false)) return {"\n\nuser: ", "\n\nassistant: "};
        const std::string marker = "TN_BOUNDARY_" + active_id + "_" + std::to_string(call_index);
        const std::string assistant_marker = marker + "_ASSISTANT";
        const std::string result_marker = marker + "_RESULT";
        json probe = history;
        probe["messages"].push_back({{"role", "assistant"}, {"content", assistant_marker}});
        probe["messages"].push_back({{"role", "user"}, {"content", result_marker}});
        const std::string formatted = render(probe);
        const size_t a = formatted.find(assistant_marker), b = formatted.find(result_marker);
        require(a != std::string::npos && b != std::string::npos && b >= a + assistant_marker.size(),
                "chat template cannot frame tool continuation");
        require(formatted.find(assistant_marker, a + 1) == std::string::npos &&
                formatted.find(result_marker, b + 1) == std::string::npos,
                "ambiguous tool continuation markers");
        return {formatted.substr(a + assistant_marker.size(), b - a - assistant_marker.size()),
                formatted.substr(b + result_marker.size())};
    }

    void decode(std::vector<llama_token> & tokens, size_t start, size_t count) {
        auto batch = llama_batch_get_one(tokens.data() + start, static_cast<int32_t>(count));
        int result = llama_decode(context, batch);
        llama_synchronize(context);
        require(capture.error.empty(), capture.error);
        require(result == 0, "decode failed (code " + std::to_string(result) + "); reduce context/batch or free memory");
    }

    json settings(const json & command) {
        const auto * vocab = llama_model_get_vocab(model);
        const bool raw = command.value("raw", false);
        const auto & messages = command.at("messages");
        const bool verbatim = raw && messages.size() == 1 && messages.at(0).at("role") == "user";
        return {{"mode", raw ? "raw" : "chat"},
                {"raw_format", raw ? json(verbatim ? "single-user-verbatim-v1" : "role-labeled-dialogue-v1") : json(nullptr)},
                {"enable_thinking", raw ? json(nullptr) : json(false)}, {"add_generation_prompt", !verbatim},
                {"template_time_unix", raw ? json(nullptr) : json(0)}, {"add_bos", llama_vocab_get_add_bos(vocab)},
                {"add_eos", false}, {"parse_special", true}, {"capture", "last-assistant-content-token"}};
    }

    void extract(const json & command, bool probe) {
        ensure_loaded();
        reset();
        std::string prefix = render(command);
        std::string completion = command.at("completion").get<std::string>();
        require(!completion.empty(), "assistant completion is empty");
        std::string rendered = prefix + completion;
        auto tokens = tokenize(rendered);
        const auto * vocab = llama_model_get_vocab(model);
        require(!llama_vocab_is_eog(vocab, tokens.back()) && !llama_vocab_is_control(vocab, tokens.back()),
                "final token is a control/end marker, not assistant content");
        capture.layers.clear();
        for (int layer : command.at("layers")) {
            require(layer >= 1 && layer < layers, "unsupported capture layer");
            require(capture.layers.insert(layer).second, "duplicate capture layer");
        }
        require(!capture.layers.empty(), "no capture layers requested");
        int chunk = command.value("prefix_chunk", batch_size);
        require(chunk >= 1 && chunk <= batch_size, "invalid prefix chunk size");
        for (size_t i = 0; i + 1 < tokens.size();) {
            boundary(0, false);
            if (cancelled) { emit(active_id, "done", {{"cancelled", true}}); return; }
            size_t n = std::min(static_cast<size_t>(chunk), tokens.size() - 1 - i);
            decode(tokens, i, n);
            i += n;
        }
        if (probe && command.contains("controls")) apply(command.at("controls"), 0, false);
        capture.active = true;
        decode(tokens, tokens.size() - 1, 1);
        capture.active = false;
        json captures = json::array();
        for (int layer : capture.layers) {
            require(capture.values.count(layer), "missing exact l_out-" + std::to_string(layer) + " capture; unsupported hook");
            const auto & values = capture.values.at(layer);
            double squared = 0;
            for (float value : values) squared += double(value) * value;
            captures.push_back({{"layer", layer}, {"values", values}, {"norm", std::sqrt(squared)}});
        }
        json result = {{"prefix", prefix}, {"rendered", rendered}, {"token_ids", tokens},
                       {"capture_position", tokens.size() - 1}, {"captures", captures},
                       {"template", command.value("raw", false) ? "" : common_chat_templates_source(templates.get())},
                       {"settings", settings(command)}, {"runtime", runtime()}};
        if (probe && command.value("include_logits", false)) {
            auto * logits = llama_get_logits_ith(context, -1);
            require(logits != nullptr, "missing logits");
            int n = llama_vocab_n_tokens(llama_model_get_vocab(model));
            result["logits"] = std::vector<float>(logits, logits + n);
        }
        emit(active_id, "result", std::move(result));
    }

    std::string complete_utf8(const std::string & piece, bool flush = false) {
        pending_utf8 += piece;
        size_t at = 0;
        while (at < pending_utf8.size()) {
            unsigned char c = pending_utf8[at];
            size_t length = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            if (at + length > pending_utf8.size() && !flush) break;
            at = std::min(at + length, pending_utf8.size());
        }
        std::string result = pending_utf8.substr(0, at);
        pending_utf8.erase(0, at);
        return result;
    }

    void generate(const json & command) {
        ensure_loaded();
        reset();
        awaiting_tool.clear();
        tool_reply = nullptr;
        auto sampling = command.value("sampling", json::object());
        const bool unbounded = sampling.value("unbounded", false);
        const bool tools = command.value("tools_enabled", false);
        const size_t maximum = unbounded ? std::numeric_limits<size_t>::max() : sampling.value("max_tokens", size_t(256));
        const size_t capacity = llama_n_ctx(context);
        std::string rendered = render(command);
        auto tokens = tokenize(rendered, unbounded);
        float temperature = sampling.value("temperature", 0.7f);
        float top_p = sampling.value("top_p", 0.95f);
        uint32_t seed = sampling.value("seed", 42u);
        require(unbounded || (maximum >= 1 && maximum <= 32768), "max_tokens must be 1..32768");
        require(unbounded || tokens.size() + maximum <= capacity, "input plus output exceeds context capacity; enable Unbounded output or reduce length");
        require(std::isfinite(temperature) && temperature >= 0 && temperature <= 100, "invalid temperature");
        require(std::isfinite(top_p) && top_p > 0 && top_p <= 1, "invalid top_p");
        apply(command.value("controls", json::object()), 0);
        // Recurrent and attention memory are both rebuilt on rollover. No unsupported
        // partial KV deletion; the original prefix plus recent tail is explicit.
        const size_t prefix = std::min(tokens.size(), capacity / 4);
        auto compact = [&](size_t first_token) {
            size_t recent = std::min(tokens.size() - prefix, capacity / 2);
            size_t dropped = tokens.size() - prefix - recent;
            std::vector<llama_token> kept(tokens.begin(), tokens.begin() + prefix);
            kept.insert(kept.end(), tokens.end() - recent, tokens.end());
            tokens.swap(kept);
            emit(active_id, "context_rollover", {{"first_token_index", first_token},
                {"retained_prefix_tokens", prefix}, {"retained_recent_tokens", recent}, {"dropped_tokens", dropped},
                {"method", "clear-all-memory-and-replay-prefix-plus-recent-tail"}});
        };
        if (tokens.size() >= capacity) compact(0);
        emit(active_id, "rendered", {{"rendered", rendered}, {"token_ids", tokens},
             {"template", command.value("raw", false) ? "" : common_chat_templates_source(templates.get())},
             {"settings", settings(command)}, {"tools_enabled", tools}, {"unbounded", unbounded}, {"runtime", runtime()}});
        auto prefill = [&](size_t first_token) {
            for (size_t i = 0; i < tokens.size() && !cancelled;) {
                boundary(first_token, true);
                if (cancelled) break;
                size_t n = std::min(static_cast<size_t>(batch_size), tokens.size() - i);
                decode(tokens, i, n);
                i += n;
            }
        };
        prefill(0);
        auto append = [&](const std::vector<llama_token> & added, size_t first_token) {
            for (size_t offset = 0; offset < added.size() && !cancelled;) {
                boundary(first_token, true);
                if (cancelled) break;
                if (tokens.size() == capacity) {
                    require(unbounded || tools, "context capacity reached");
                    compact(first_token);
                    llama_synchronize(context);
                    llama_memory_clear(llama_get_memory(context), true);
                    prefill(first_token);
                    if (cancelled) break;
                }
                size_t n = std::min({static_cast<size_t>(batch_size), capacity - tokens.size(), added.size() - offset});
                size_t start = tokens.size();
                tokens.insert(tokens.end(), added.begin() + offset, added.begin() + offset + n);
                decode(tokens, start, n);
                offset += n;
            }
        };
        using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
        Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        if (temperature == 0) llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        else {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));
        }
        std::string output, pending, assistant_segment;
        json reply_messages = json::array();
        json tool_history = command;
        pending_utf8.clear();
        size_t produced = 0, tool_count = 0;
        bool in_tool = false;
        const std::string open = "<torment_tool>", close = "</torment_tool>";
        const auto * vocab = llama_model_get_vocab(model);
        for (; produced < maximum && !cancelled; ++produced) {
            // Logits already exist. Updates here first affect the next distribution.
            llama_token token = llama_sampler_sample(sampler.get(), context, -1);
            if (llama_vocab_is_eog(vocab, token)) break;
            std::string piece = common_token_to_piece(vocab, token, false);
            assistant_segment += piece;
            std::string visible = complete_utf8(piece);
            std::optional<std::string> call;
            if (tools) {
                pending += visible;
                visible.clear();
                if (!in_tool) {
                    const size_t at = pending.find(open);
                    if (at != std::string::npos) {
                        visible = pending.substr(0, at);
                        pending.erase(0, at + open.size());
                        in_tool = true;
                    } else {
                        size_t keep = std::min(pending.size(), open.size() - 1);
                        while (keep && pending.compare(pending.size() - keep, keep, open, 0, keep) != 0) --keep;
                        visible = pending.substr(0, pending.size() - keep);
                        pending.erase(0, pending.size() - keep);
                    }
                }
                if (in_tool) {
                    const size_t at = pending.find(close);
                    require(pending.size() <= 16384, "tool call exceeds 16 KiB or is missing its closing tag");
                    if (at != std::string::npos) {
                        call = pending.substr(0, at);
                        pending.erase(0, at + close.size());
                        in_tool = false;
                    }
                }
                output += visible;
            } else output += piece;
            emit(active_id, "token", {{"text", visible}, {"index", produced}, {"token", token}, {"revision", revision}});
            boundary(produced + 1, unbounded || produced + 1 < maximum || call.has_value());
            if (cancelled) { ++produced; break; }
            if (call) {
                awaiting_tool = std::to_string(++tool_count);
                tool_reply = nullptr;
                emit(active_id, "tool_call", {{"tool_id", awaiting_tool}, {"json", *call}, {"after_token_index", produced}});
                while (tool_reply.is_null() && !cancelled) {
                    boundary(produced + 1, true);
                    if (tool_reply.is_null() && !cancelled) {
                        std::unique_lock lock(inbox.mutex);
                        inbox.ready.wait_for(lock, std::chrono::milliseconds(100), [&] { return !inbox.commands.empty() || inbox.eof; });
                    }
                }
                awaiting_tool.clear();
                if (cancelled) { ++produced; break; }
            }
            const bool can_continue = unbounded || produced + 1 < maximum;
            if (can_continue) append({token}, produced + 1);
            if (call && !cancelled) {
                std::string feedback = "Torment Nexus tool result:\n<torment_result>" + tool_reply.dump() +
                    "</torment_result>\nContinue your response using this result; you may call another tool if needed.";
                if (can_continue) {
                    const auto framed = tool_boundaries(tool_history, tool_count);
                    append(common_tokenize(vocab, framed.first, false, true), produced + 1);
                    append(common_tokenize(vocab, feedback, false, false), produced + 1);
                    append(common_tokenize(vocab, framed.second, false, true), produced + 1);
                    emit(active_id, "tool_continuation", {{"tool_id", std::to_string(tool_count)},
                        {"first_token_index", produced + 1}, {"before_result", framed.first}, {"after_result", framed.second}});
                }
                const json assistant = {{"role", "assistant"}, {"content", assistant_segment}};
                const json result_message = {{"role", "user"}, {"content", feedback}};
                reply_messages.push_back(assistant);
                reply_messages.push_back(result_message);
                tool_history["messages"].push_back(assistant);
                tool_history["messages"].push_back(result_message);
                assistant_segment.clear();
            }
        }
        // Incomplete markup is retained as text, never executed.
        if (tools) output += (in_tool ? open : "") + pending;
        reply_messages.push_back({{"role", "assistant"}, {"content", assistant_segment}});
        emit(active_id, "done", {{"output", output}, {"cancelled", cancelled},
                                 {"tokens", produced}, {"revision", revision}, {"runtime", runtime()},
                                 {"reply_messages", reply_messages}});
    }

    void unload() {
        templates.reset();
        if (context) { llama_free(context); context = nullptr; }
        if (model) { llama_model_free(model); model = nullptr; }
        width = layers = 0;
        model_path.clear();
    }

    void load(const json & command) {
        unload();
        model_path = command.at("path").get<std::string>();
        int context_size = command.value("context", 8192);
        batch_size = command.value("batch", 512);
        int microbatch = command.value("microbatch", 128);
        require(context_size >= 128 && context_size <= 262144, "context must be 128..262144");
        require(batch_size >= 1 && batch_size <= 4096 && microbatch >= 1 && microbatch <= batch_size, "invalid batch sizes");
        auto params = llama_model_default_params();
        params.n_gpu_layers = command.value("gpu_layers", 99);
        model = llama_model_load_from_file(model_path.c_str(), params);
        require(model != nullptr, "model load failed: verify GGUF/runtime compatibility and available memory; see worker log");
        try {
            width = llama_model_n_embd(model);
            layers = llama_model_n_layer(model);
            require(width > 0 && width <= 65536 && layers > 1 && layers <= 1024, "unsupported model shape");
            capture.width = width;
            auto ctx = llama_context_default_params();
            ctx.n_ctx = context_size;
            ctx.n_batch = batch_size;
            ctx.n_ubatch = microbatch;
            ctx.n_seq_max = 1;
            ctx.type_k = GGML_TYPE_F16;
            ctx.type_v = GGML_TYPE_F16;
            ctx.cb_eval = Capture::callback;
            ctx.cb_eval_user_data = &capture;
            context = llama_init_from_model(model, ctx);
            require(context != nullptr, "context allocation failed: lower context or batch sizes, or free memory");
            const char * source = llama_model_chat_template(model, nullptr);
            if (source && *source) templates = common_chat_templates_init(model, "");
            json supported = json::array();
            for (int i = 1; i < layers; ++i) supported.push_back(i);
            emit(active_id, "result", {{"path", model_path}, {"n_layer", layers}, {"n_embd", width},
                 {"context", llama_n_ctx(context)}, {"batch", batch_size}, {"microbatch", microbatch},
                 {"supported_layers", supported}, {"chat_template", source ? source : ""},
                 {"self_tools", true}, {"unbounded_output", true},
                 {"model_bytes", llama_model_size(model)}, {"runtime", runtime()}});
        } catch (...) { unload(); throw; }
    }

public:
    explicit Engine(Inbox & inbox) : inbox(inbox) {}
    ~Engine() { unload(); }
    void run() {
        while (true) {
            auto command = inbox.next();
            if (command.is_null()) break;
            active_id = command.value("id", "invalid");
            try {
                require(command.value("v", 0) == protocol, "unsupported protocol version");
                auto op = command.at("op").get<std::string>();
                if (op == "capabilities") emit(active_id, "result", {{"runtime", runtime()},
                    {"capture", "exact l_out-{graph_layer}, contiguous F32, final token only"},
                    {"steering_layers", "1..n_layer-1"}, {"live_controls", true}, {"self_tools", true}, {"unbounded_output", true}, {"cache", "F16"}});
                else if (op == "load") load(command);
                else if (op == "unload") { unload(); emit(active_id, "result"); }
                else if (op == "extract") extract(command, false);
                else if (op == "probe") extract(command, true); // private engine conformance tests
                else if (op == "generate") generate(command);
                else if (op == "cancel") emit(active_id, "result", {{"cancel_requested", false}});
                else throw std::runtime_error("unknown or inactive operation: " + op);
            } catch (const std::bad_alloc &) {
                capture.active = false;
                emit(active_id, "error", {{"error", "allocation failed; lower context/batch or unload model"}});
            } catch (const std::exception & e) {
                capture.active = false;
                emit(active_id, "error", {{"error", e.what()}});
            }
        }
    }
};

int main() {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set([](ggml_log_level, const char * text, void *) { std::fputs(text, stderr); }, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    Inbox inbox;
    std::thread reader([&] { inbox.read(); });
    { Engine engine(inbox); engine.run(); }
    reader.join();
    llama_backend_free();
}
