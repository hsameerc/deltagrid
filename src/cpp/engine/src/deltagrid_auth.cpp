#ifndef DELTAGRID_EXPORTS
#define DELTAGRID_EXPORTS
#endif

#include "deltagrid_internal.h"
#include <string.h>

#ifdef USE_DRM
#include "drm.h"
bool is_deltagrid_unlocked = false;
#else
bool is_deltagrid_unlocked = true;
#endif

extern "C" {

DELTAGRID_API bool deltagrid_authenticate(const char* hardware_id) {
#ifndef USE_DRM
    (void)hardware_id;
    is_deltagrid_unlocked = true;
    return true;
#else
    if (hardware_id == nullptr) return false;

    // Compute FNV-1a 32-bit hash
    uint32_t hash_val = 2166136261u;
    size_t len = strlen(hardware_id);
    for (size_t i = 0; i < len; ++i) {
        hash_val ^= (uint8_t)hardware_id[i];
        hash_val *= 16777619u;
    }

    for (size_t i = 0; i < NUM_AUTHORIZED_HWIDS; ++i) {
        if (hash_val == AUTHORIZED_HWID_HASHES[i]) {
            is_deltagrid_unlocked = true;
            return true;
        }
    }
    return false;
#endif
}

}

