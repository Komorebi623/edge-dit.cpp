#include "light-dit.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> --prompt <text> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> --t5xxl <path> --prompt <text> [options]\n\n"
        "Options:\n"
        "  --video                   Generate video frames instead of an image\n"
        "  --video-format <fmt>      Video format: auto, avi, mp4, mov, mkv, webm. Default: auto\n"
        "  --diffusion-model <path>  Standalone Flux transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  -o, --output <path>       Output image/video path, default: output.png\n"
        "  -W, --width <int>         Image width, default: 1024\n"
        "  -H, --height <int>        Image height, default: 1024\n"
        "  --frames <int>            Video frame count, default: 1\n"
        "  --fps <int>               Video fps, default: 16\n"
        "  --steps <int>             Sampling steps, default: 20\n"
        "  -s, --seed <int64>        Seed, default: -1\n"
        "  -t, --threads <int>       Thread count, default: 0\n"
        "  --guidance <float>        Flux distilled guidance, default: 3.5\n"
        "  --backend <name>          Backend: auto, cpu, cuda, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  --help              Show this help\n",
        prog,
        prog
    );
}

static bool save_png(const char* path, const ld_image_t& image) {
    if (path == nullptr || image.data == nullptr) {
        return false;
    }

    if (image.channels == 0 || image.channels > 4) {
        std::fprintf(stderr, "unsupported channel count: %u\n", image.channels);
        return false;
    }

    return stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channels),
        image.data,
        0,
        nullptr
    ) != 0;
}

static std::string shell_quote(const char* value) {
    std::string quoted = "'";
    const char* text = value != nullptr ? value : "";
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == '\'') {
            quoted += "'\\''";
        } else {
            quoted += *p;
        }
    }
    quoted += "'";
    return quoted;
}

static std::string lowercase(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return value;
}

static std::string path_extension(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return "";
    }
    return lowercase(path.substr(dot));
}

static std::string replace_extension(const std::string& path, const std::string& ext) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + ext;
    }
    return path.substr(0, dot) + ext;
}

static bool is_ffmpeg_video_ext(const std::string& ext) {
    return ext == ".mp4" || ext == ".mov" || ext == ".mkv" || ext == ".webm";
}

static bool is_supported_video_ext(const std::string& ext) {
    return ext == ".avi" || is_ffmpeg_video_ext(ext);
}

static std::string normalized_video_format(const char* format) {
    std::string value = format != nullptr && format[0] != '\0' ? format : "auto";
    value = lowercase(value);
    if (!value.empty() && value[0] == '.') {
        value.erase(value.begin());
    }
    return value;
}

static std::string video_output_path(const char* output_path, const char* format) {
    std::string path = output_path != nullptr && output_path[0] != '\0' ? output_path : "output.avi";
    const std::string requested_format = normalized_video_format(format);

    if (requested_format != "auto") {
        return replace_extension(path, "." + requested_format);
    }

    const std::string ext = path_extension(path);
    if (!is_supported_video_ext(ext)) {
        path = replace_extension(path, ".avi");
    }
    return path;
}

static std::string find_imageio_ffmpeg_in_conda(const char* conda_prefix) {
    if (conda_prefix == nullptr || conda_prefix[0] == '\0') {
        return "";
    }

    const fs::path lib_dir = fs::path(conda_prefix) / "lib";
    std::error_code ec;
    if (!fs::is_directory(lib_dir, ec)) {
        return "";
    }

    for (const fs::directory_entry& python_entry : fs::directory_iterator(lib_dir, ec)) {
        if (ec) {
            break;
        }
        if (!python_entry.is_directory()) {
            continue;
        }
        const std::string python_dir_name = python_entry.path().filename().string();
        if (python_dir_name.rfind("python", 0) != 0) {
            continue;
        }

        const fs::path binaries_dir = python_entry.path() / "site-packages" / "imageio_ffmpeg" / "binaries";
        std::error_code bin_ec;
        if (!fs::is_directory(binaries_dir, bin_ec)) {
            continue;
        }
        for (const fs::directory_entry& ffmpeg_entry : fs::directory_iterator(binaries_dir, bin_ec)) {
            if (bin_ec) {
                break;
            }
            const std::string name = ffmpeg_entry.path().filename().string();
            if (ffmpeg_entry.is_regular_file() && name.rfind("ffmpeg-", 0) == 0) {
                return ffmpeg_entry.path().string();
            }
        }
    }
    return "";
}

static std::string find_ffmpeg_binary() {
    const char* configured = std::getenv("LDIT_FFMPEG");
    if (configured != nullptr && configured[0] != '\0') {
        return configured;
    }

    std::string bundled = find_imageio_ffmpeg_in_conda(std::getenv("CONDA_PREFIX"));
    if (!bundled.empty()) {
        return bundled;
    }

    bundled = find_imageio_ffmpeg_in_conda("/export/home/liuyiming54/miniconda3/envs/hicache");
    if (!bundled.empty()) {
        return bundled;
    }

    return "ffmpeg";
}

static bool write_rgb_frame(FILE* pipe, const ld_image_t& image, std::vector<uint8_t>* scratch) {
    if (pipe == nullptr || image.data == nullptr || scratch == nullptr) {
        return false;
    }

    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    if (image.channels == 3) {
        return std::fwrite(image.data, 1, pixels * 3, pipe) == pixels * 3;
    }

    if (image.channels != 1 && image.channels != 4) {
        std::fprintf(stderr, "unsupported video frame channel count: %u\n", image.channels);
        return false;
    }

    scratch->resize(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        if (image.channels == 1) {
            const uint8_t v = image.data[i];
            (*scratch)[i * 3 + 0] = v;
            (*scratch)[i * 3 + 1] = v;
            (*scratch)[i * 3 + 2] = v;
        } else {
            (*scratch)[i * 3 + 0] = image.data[i * 4 + 0];
            (*scratch)[i * 3 + 1] = image.data[i * 4 + 1];
            (*scratch)[i * 3 + 2] = image.data[i * 4 + 2];
        }
    }
    return std::fwrite(scratch->data(), 1, scratch->size(), pipe) == scratch->size();
}

static void write_u16_le(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

static void write_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

static void patch_u32_le(std::vector<uint8_t>& out, size_t pos, uint32_t value) {
    out[pos + 0] = static_cast<uint8_t>(value & 0xff);
    out[pos + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
    out[pos + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
    out[pos + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

static void write_fourcc(std::vector<uint8_t>& out, const char* fourcc) {
    out.insert(out.end(), fourcc, fourcc + 4);
}

static bool write_binary_file(const char* path, const std::vector<uint8_t>& data) {
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "failed to open output file: %s\n", path);
        return false;
    }
    const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
    std::fclose(file);
    return ok;
}

static bool image_to_rgb(const ld_image_t& image, std::vector<uint8_t>* rgb) {
    if (image.data == nullptr || rgb == nullptr || image.width == 0 || image.height == 0) {
        return false;
    }

    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    rgb->resize(pixels * 3);

    if (image.channels == 3) {
        std::memcpy(rgb->data(), image.data, rgb->size());
        return true;
    }
    if (image.channels == 4) {
        for (size_t i = 0; i < pixels; ++i) {
            (*rgb)[i * 3 + 0] = image.data[i * 4 + 0];
            (*rgb)[i * 3 + 1] = image.data[i * 4 + 1];
            (*rgb)[i * 3 + 2] = image.data[i * 4 + 2];
        }
        return true;
    }
    if (image.channels == 1) {
        for (size_t i = 0; i < pixels; ++i) {
            const uint8_t v = image.data[i];
            (*rgb)[i * 3 + 0] = v;
            (*rgb)[i * 3 + 1] = v;
            (*rgb)[i * 3 + 2] = v;
        }
        return true;
    }

    std::fprintf(stderr, "unsupported video frame channel count: %u\n", image.channels);
    return false;
}

struct AviIndexEntry {
    char fourcc[4];
    uint32_t flags = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
};

static bool save_mjpg_avi(const char* path, const ld_video_t& video, int fps, int quality) {
    if (path == nullptr || video.frames == nullptr || video.frame_count <= 0 || fps <= 0) {
        return false;
    }

    const ld_image_t& first = video.frames[0];
    if (first.data == nullptr || first.width == 0 || first.height == 0) {
        return false;
    }

    const uint32_t width = first.width;
    const uint32_t height = first.height;
    const uint32_t frame_count = static_cast<uint32_t>(video.frame_count);
    const int jpg_quality = quality < 1 ? 1 : (quality > 100 ? 100 : quality);

    std::vector<uint8_t> avi;
    avi.reserve(static_cast<size_t>(width) * height * 3 * video.frame_count / 4);

    write_fourcc(avi, "RIFF");
    const size_t riff_size_pos = avi.size();
    write_u32_le(avi, 0);
    write_fourcc(avi, "AVI ");

    write_fourcc(avi, "LIST");
    write_u32_le(avi, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi, "hdrl");

    write_fourcc(avi, "avih");
    write_u32_le(avi, 56);
    write_u32_le(avi, static_cast<uint32_t>(1000000 / fps));
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0x110);
    write_u32_le(avi, frame_count);
    write_u32_le(avi, 0);
    write_u32_le(avi, 1);
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, width);
    write_u32_le(avi, height);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);

    write_fourcc(avi, "LIST");
    write_u32_le(avi, 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi, "strl");

    write_fourcc(avi, "strh");
    write_u32_le(avi, 56);
    write_fourcc(avi, "vids");
    write_fourcc(avi, "MJPG");
    write_u32_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 1);
    write_u32_le(avi, static_cast<uint32_t>(fps));
    write_u32_le(avi, 0);
    write_u32_le(avi, frame_count);
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, 0xffffffffu);
    write_u32_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);
    write_u16_le(avi, 0);

    write_fourcc(avi, "strf");
    write_u32_le(avi, 40);
    write_u32_le(avi, 40);
    write_u32_le(avi, width);
    write_u32_le(avi, height);
    write_u16_le(avi, 1);
    write_u16_le(avi, 24);
    write_fourcc(avi, "MJPG");
    write_u32_le(avi, width * height * 3);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);
    write_u32_le(avi, 0);

    write_fourcc(avi, "LIST");
    const size_t movi_size_pos = avi.size();
    write_u32_le(avi, 0);
    write_fourcc(avi, "movi");

    std::vector<AviIndexEntry> index;
    index.reserve(static_cast<size_t>(video.frame_count));
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> jpg;

    for (int i = 0; i < video.frame_count; ++i) {
        const ld_image_t& frame = video.frames[i];
        if (frame.width != width || frame.height != height || !image_to_rgb(frame, &rgb)) {
            std::fprintf(stderr, "video frame %d has invalid or inconsistent data\n", i);
            return false;
        }

        jpg.clear();
        auto write_jpg = [](void* context, void* data, int size) {
            auto* buffer = static_cast<std::vector<uint8_t>*>(context);
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            buffer->insert(buffer->end(), bytes, bytes + size);
        };
        if (!stbi_write_jpg_to_func(write_jpg,
                                    &jpg,
                                    static_cast<int>(width),
                                    static_cast<int>(height),
                                    3,
                                    rgb.data(),
                                    jpg_quality)) {
            std::fprintf(stderr, "failed to encode AVI frame %d as JPEG\n", i);
            return false;
        }

        AviIndexEntry entry{};
        std::memcpy(entry.fourcc, "00dc", 4);
        entry.flags = 0x10;
        entry.offset = static_cast<uint32_t>(avi.size());
        entry.size = static_cast<uint32_t>(jpg.size());

        write_fourcc(avi, "00dc");
        write_u32_le(avi, entry.size);
        avi.insert(avi.end(), jpg.begin(), jpg.end());
        if (jpg.size() % 2 != 0) {
            avi.push_back(0);
        }
        index.push_back(entry);
    }

    patch_u32_le(avi, movi_size_pos, static_cast<uint32_t>(avi.size() - movi_size_pos - 4));

    write_fourcc(avi, "idx1");
    write_u32_le(avi, static_cast<uint32_t>(index.size() * 16));
    for (const AviIndexEntry& entry : index) {
        write_fourcc(avi, entry.fourcc);
        write_u32_le(avi, entry.flags);
        write_u32_le(avi, entry.offset);
        write_u32_le(avi, entry.size);
    }

    patch_u32_le(avi, riff_size_pos, static_cast<uint32_t>(avi.size() - riff_size_pos - 4));
    return write_binary_file(path, avi);
}

static bool save_ffmpeg_video(const char* path, const ld_video_t& video, int fps) {
    if (path == nullptr || video.frames == nullptr || video.frame_count <= 0) {
        return false;
    }
    if (fps <= 0) {
        fps = 16;
    }

    const ld_image_t& first = video.frames[0];
    if (first.data == nullptr || first.width == 0 || first.height == 0) {
        return false;
    }

    for (int i = 0; i < video.frame_count; ++i) {
        const ld_image_t& frame = video.frames[i];
        if (frame.data == nullptr || frame.width != first.width || frame.height != first.height) {
            std::fprintf(stderr, "video frame %d has inconsistent dimensions\n", i);
            return false;
        }
    }

    std::signal(SIGPIPE, SIG_IGN);

    char cmd[4096];
    const std::string ffmpeg_path = find_ffmpeg_binary();
    const std::string quoted_ffmpeg = shell_quote(ffmpeg_path.c_str());
    const std::string quoted_path = shell_quote(path);
    const std::string ext = path_extension(path);
    const char* codec_args = ext == ".webm"
                                 ? "-an -c:v libvpx-vp9 -crf 18 -b:v 0 -pix_fmt yuv420p"
                                 : "-an -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p";
    const char* mux_args = ext == ".webm" ? "" : "-movflags +faststart";
    std::snprintf(cmd,
                  sizeof(cmd),
                  "%s -hide_banner -loglevel error -y "
                  "-f rawvideo -pix_fmt rgb24 -s %ux%u -r %d -i - "
                  "%s %s %s",
                  quoted_ffmpeg.c_str(),
                  first.width,
                  first.height,
                  fps,
                  codec_args,
                  mux_args,
                  quoted_path.c_str());

    FILE* pipe = popen(cmd, "w");
    if (pipe == nullptr) {
        std::fprintf(stderr, "failed to start ffmpeg: %s\n", std::strerror(errno));
        return false;
    }

    std::vector<uint8_t> scratch;
    bool ok = true;
    for (int i = 0; i < video.frame_count; ++i) {
        if (!write_rgb_frame(pipe, video.frames[i], &scratch)) {
            ok = false;
            break;
        }
    }

    const int status = pclose(pipe);
    if (!ok || status != 0) {
        if (status == 32512) {
            std::fprintf(stderr, "ffmpeg was not found; install ffmpeg, add it to PATH, or set LDIT_FFMPEG\n");
        } else {
            std::fprintf(stderr, "ffmpeg failed while writing video, status=%d\n", status);
        }
        return false;
    }
    return true;
}

static bool save_video(const char* path, const ld_video_t& video, int fps) {
    const std::string ext = path_extension(path != nullptr ? path : "");
    if (ext == ".avi") {
        return save_mjpg_avi(path, video, fps, 95);
    }
    if (is_ffmpeg_video_ext(ext)) {
        return save_ffmpeg_video(path, video, fps);
    }
    std::fprintf(stderr, "unsupported video extension: %s\n", ext.c_str());
    return false;
}

struct FluxCliArgs {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* clip_l_path = nullptr;
    const char* t5xxl_path = nullptr;
    const char* prompt = nullptr;
    const char* output_path = "output.png";
    const char* video_format = nullptr;
    const char* backend = nullptr;

    bool video = false;
    int width = 1024;
    int height = 1024;
    int frames = 1;
    int fps = 16;
    int steps = 20;
    int threads = 0;

    int64_t seed = -1;
    float guidance = 3.5f;
};

static bool parse_args(int argc, char** argv, FluxCliArgs* args) {
    if (args == nullptr) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--video") == 0) {
            args->video = true;
        } else if (std::strcmp(key, "--video-format") == 0) {
            args->video_format = require_value(key);
        } else if (std::strcmp(key, "--model") == 0) {
            args->model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->vae_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--prompt") == 0 || std::strcmp(key, "-p") == 0) {
            args->prompt = require_value(key);
        } else if (std::strcmp(key, "--output") == 0 || std::strcmp(key, "-o") == 0) {
            args->output_path = require_value(key);
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->width = std::atoi(v);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->height = std::atoi(v);
        } else if (std::strcmp(key, "--frames") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->frames = std::atoi(v);
        } else if (std::strcmp(key, "--fps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->fps = std::atoi(v);
        } else if (std::strcmp(key, "--steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->steps = std::atoi(v);
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->threads = std::atoi(v);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->seed = std::strtoll(v, nullptr, 10);
        } else if (std::strcmp(key, "--guidance") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->guidance = std::strtof(v, nullptr);
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = args->model_path != nullptr && std::strlen(args->model_path) > 0;
    const bool has_components =
        args->diffusion_model_path != nullptr && std::strlen(args->diffusion_model_path) > 0 &&
        args->vae_path != nullptr && std::strlen(args->vae_path) > 0 &&
        args->clip_l_path != nullptr && std::strlen(args->clip_l_path) > 0 &&
        args->t5xxl_path != nullptr && std::strlen(args->t5xxl_path) > 0;
    if (!has_full_model && !has_components) {
        std::fprintf(stderr, "--model or the full --diffusion-model/--vae/--clip_l/--t5xxl set is required\n");
        return false;
    }

    if (args->prompt == nullptr || std::strlen(args->prompt) == 0) {
        std::fprintf(stderr, "--prompt is required\n");
        return false;
    }

    if (args->width <= 0 || args->height <= 0) {
        std::fprintf(stderr, "width and height must be positive\n");
        return false;
    }

    if (args->frames <= 0) {
        std::fprintf(stderr, "frames must be positive\n");
        return false;
    }

    if (args->fps <= 0) {
        std::fprintf(stderr, "fps must be positive\n");
        return false;
    }

    const std::string video_format = normalized_video_format(args->video_format);
    if (video_format != "auto" &&
        video_format != "avi" &&
        video_format != "mp4" &&
        video_format != "mov" &&
        video_format != "mkv" &&
        video_format != "webm") {
        std::fprintf(stderr, "unsupported video format: %s\n", video_format.c_str());
        return false;
    }

    if (args->steps <= 0) {
        std::fprintf(stderr, "steps must be positive\n");
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    FluxCliArgs args;

    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.backend != nullptr && std::strlen(args.backend) > 0) {
        setenv("LDIT_BACKEND", args.backend, 1);
    }

    ld_context_params_t ctx_params;
    ld_context_params_init(&ctx_params);

    ctx_params.model_path = args.model_path;
    ctx_params.diffusion_model_path = args.diffusion_model_path;
    ctx_params.vae_path = args.vae_path;
    ctx_params.clip_l_path = args.clip_l_path;
    ctx_params.t5xxl_path = args.t5xxl_path;

    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
    }

    /*
     * Flux 测试阶段先让内部自动识别 dtype / sampler / scheduler。
     * 如果你的模型是量化 GGUF，也可以在这里手动指定：
     *   ctx_params.weight_type = LD_DTYPE_Q8_0;
     */
    ctx_params.weight_type = LD_DTYPE_AUTO;

    ld_context_t* ctx = ld_create_context(&ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create light-dit context\n");
        return 2;
    }

    if (args.video) {
        ld_video_generation_params_t gen_params;
        ld_video_generation_params_init(&gen_params);

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = "";
        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.frames = args.frames;
        gen_params.seed = args.seed;
        gen_params.sample.sampler = LD_SAMPLER_AUTO;
        gen_params.sample.scheduler = LD_SCHEDULER_AUTO;
        gen_params.sample.steps = args.steps;
        gen_params.sample.cfg_scale = 1.0f;
        gen_params.sample.image_cfg_scale = 1.0f;
        gen_params.sample.distilled_guidance = args.guidance;

        ld_video_t output;
        ld_status_t status = ld_generate_video(ctx, &gen_params, &output);
        if (status != LD_STATUS_OK) {
            std::fprintf(stderr, "ld_generate_video failed, status=%d\n", static_cast<int>(status));
            const char* err = ld_get_last_error(ctx);
            if (err != nullptr && std::strlen(err) > 0) {
                std::fprintf(stderr, "last error: %s\n", err);
            }
            ld_free_context(ctx);
            return 3;
        }

        if (output.frame_count <= 0 || output.frames == nullptr) {
            std::fprintf(stderr, "generation succeeded but video output is empty\n");
            ld_free_context(ctx);
            return 4;
        }

        const std::string output_path = video_output_path(args.output_path, args.video_format);
        if (!save_video(output_path.c_str(), output, args.fps)) {
            std::fprintf(stderr, "failed to save output video: %s\n", output_path.c_str());
            ld_free_video(&output);
            ld_free_context(ctx);
            return 5;
        }

        std::printf("saved video to %s\n", output_path.c_str());

        ld_free_video(&output);
    } else {
        ld_image_generation_params_t gen_params;
        ld_image_generation_params_init(&gen_params);

        gen_params.prompt = args.prompt;
        gen_params.negative_prompt = "";

        gen_params.width = args.width;
        gen_params.height = args.height;
        gen_params.seed = args.seed;
        gen_params.batch_count = 1;

        /*
         * Flux / DiT 测试建议：
         * - sampler/scheduler 用 AUTO，让内部根据模型选择
         * - cfg_scale 设为 1.0，避免传统 CFG 负条件分支
         * - distilled_guidance 用 Flux 常见值 3.5
         */
        gen_params.sample.sampler = LD_SAMPLER_AUTO;
        gen_params.sample.scheduler = LD_SCHEDULER_AUTO;
        gen_params.sample.steps = args.steps;
        gen_params.sample.cfg_scale = 1.0f;
        gen_params.sample.image_cfg_scale = 1.0f;
        gen_params.sample.distilled_guidance = args.guidance;

        ld_image_batch_t output;
        ld_status_t status = ld_generate_image(ctx, &gen_params, &output);

        if (status != LD_STATUS_OK) {
            std::fprintf(stderr, "ld_generate_image failed, status=%d\n", static_cast<int>(status));

            const char* err = ld_get_last_error(ctx);
            if (err != nullptr && std::strlen(err) > 0) {
                std::fprintf(stderr, "last error: %s\n", err);
            }

            ld_free_context(ctx);
            return 3;
        }

        if (output.count <= 0 || output.images == nullptr) {
            std::fprintf(stderr, "generation succeeded but output is empty\n");
            ld_free_context(ctx);
            return 4;
        }

        if (!save_png(args.output_path, output.images[0])) {
            std::fprintf(stderr, "failed to save output image: %s\n", args.output_path);
            ld_free_image_batch(&output);
            ld_free_context(ctx);
            return 5;
        }

        std::printf("saved image to %s\n", args.output_path);

        ld_free_image_batch(&output);
    }
    ld_free_context(ctx);

    return 0;
}
