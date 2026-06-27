#include "voicedetect.h"
#include "voicedetect_capi.h"
#include <cstdio>
#include <cstring>

// Model-independent smoke test: the version string is non-empty and the C-API
// ABI version is the expected initial value. Loads no model, returns 0.
int main() {
    const char* v = voicedetect_version();
    if (v == nullptr || std::strlen(v) == 0) {
        std::fprintf(stderr, "version string is empty\n");
        return 1;
    }
    const int abi = voicedetect_capi_abi_version();
    if (abi < 1) {
        std::fprintf(stderr, "unexpected abi version: %d\n", abi);
        return 1;
    }
    std::printf("voice-detect.cpp version: %s (C-ABI v%d)\n", v, abi);
    return 0;
}
