#include "audio_frontend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s input.pcm cmvn.bin output.bin\n", argv[0]);
    return 2;
  }
  FILE* cmvn_file = std::fopen(argv[2], "rb");
  FILE* audio_file = std::fopen(argv[1], "rb");
  FILE* output_file = std::fopen(argv[3], "wb");
  if (!cmvn_file || !audio_file || !output_file) {
    std::fprintf(stderr, "cannot open input or output file\n");
    return 3;
  }
  std::vector<float> cmvn(1120);
  if (std::fread(cmvn.data(), sizeof(float), cmvn.size(), cmvn_file) != cmvn.size()) {
    std::fprintf(stderr, "CMVN file is truncated\n");
    return 4;
  }
  std::fclose(cmvn_file);
  paraformer::AudioFrontend frontend(cmvn);
  std::array<int16_t, 9600> pcm;
  size_t frames = 0;
  while (true) {
    const size_t count = std::fread(pcm.data(), sizeof(int16_t), pcm.size(), audio_file);
    std::vector<float> audio(count);
    for (size_t i = 0; i < count; ++i) {
      audio[i] = pcm[i] / 32768.0f;
    }
    auto features = frontend.Accept(audio.data(), count, count < pcm.size());
    std::printf("block %zu features %zu\n", frames / 10, features.size());
    for (const auto& row : features) {
      std::fwrite(row.data(), sizeof(float), row.size(), output_file);
    }
    frames += features.size();
    if (count < pcm.size()) {
      break;
    }
  }
  std::fclose(audio_file);
  std::fclose(output_file);
  std::printf("frames %zu\n", frames);
  return 0;
}
