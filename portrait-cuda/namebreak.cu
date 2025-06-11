#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <string>

#define ALPHABET_SIZE 49
#define MAX_FILENAME_LEN 128
#define MAX_MATCHES 1024

__device__ volatile int d_foundMatchFlag = 0;

__device__ __constant__ char d_alphabet[ALPHABET_SIZE + 1] = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";

__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ char d_lowerBound[64];
__device__ __constant__ char d_upperBound[64];
__device__ __constant__ char lowerBound[64];
__device__ __constant__ char upperBound[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;

__device__ __constant__ uint32_t d_cryptTable[0x500];

__device__ uint32_t mpqHash(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ uint32_t mpqHashSeed2(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x200 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ void indexToZ(uint64_t index, int zLen, char* outZ) {
    for (int i = zLen - 1; i >= 0; --i) {
        outZ[i] = d_alphabet[index % ALPHABET_SIZE];
        index /= ALPHABET_SIZE;
    }
}

uint64_t stringToIndex(const std::string& str) {
    uint64_t index = 0;
    for (char c : str) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid character in string: '%c'\n", c);
            exit(1);
        }
        index = index * alphabet.size() + pos;
    }
    return index;
}

__device__ void buildCandidate(const char* Z, int zLen, int split, char* out) {
    memcpy(out, d_prefix, d_prefix_size);
    short i = d_prefix_size;

    memcpy(out + i, Z, split);
    i += split;

    out[i++] = '\\';

    memcpy(out + i, Z + split, zLen - split);
    i += zLen - split;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

__device__ void buildCandidateWithoutBackslash(const char* Z, int zLen, char* out) {
    short i = 0;
    memcpy(out + i, d_prefix, d_prefix_size);
    i += d_prefix_size;

    memcpy(out + i, Z, zLen);
    i += zLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

__device__ int my_strcmp (const char * s1, const char * s2) {
    for(; *s1 == *s2; ++s1, ++s2)
        if(*s1 == 0)
            return 0;
    return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}

__device__ bool inBounds(const char* z) {
    return my_strcmp(z, lowerBound) >= 0 &&
           my_strcmp(z, upperBound) <= 0;
}

__global__ void bruteForceKernel(
        int zLen,
        uint64_t startIdx,
        uint64_t total,
        uint32_t targetA,
        uint32_t targetB,
        char* d_matches,
        int* d_matchCount
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    idx += startIdx;

    char Z[16];
    char candidate[MAX_FILENAME_LEN];

    indexToZ(idx, zLen, Z);

    ///
    buildCandidateWithoutBackslash(Z, zLen, candidate);
    uint32_t hashA = mpqHash(candidate);
    if (hashA == targetA) {
        printf("Hash A matches: %s\n", candidate);

        uint32_t hashB = mpqHashSeed2(candidate);
        if (hashB == targetB) {
            printf("BOTH HASHES MATCH: %s\n", candidate);
            d_foundMatchFlag = 1;
        }
        int slot = atomicAdd(d_matchCount, 1);
        if (slot < MAX_MATCHES) {
            memcpy(&d_matches[slot * MAX_FILENAME_LEN], candidate, MAX_FILENAME_LEN);
        }
    }
    ///

    for (int split = 1; split <= zLen; ++split) {
        buildCandidate(Z, zLen, split, candidate);

        uint32_t hashA = mpqHash(candidate);
        if (hashA == targetA) {
            printf("Hash A matches: %s\n", candidate);

            uint32_t hashB = mpqHashSeed2(candidate);
            if (hashB == targetB) {
                printf("BOTH HASHES MATCH: %s\n", candidate);
                d_foundMatchFlag = 1;
            }
            int slot = atomicAdd(d_matchCount, 1);
            if (slot < MAX_MATCHES) {
                memcpy(&d_matches[slot * MAX_FILENAME_LEN], candidate, MAX_FILENAME_LEN);
            }
        }
    }
}

void prepareCryptTable(uint32_t* table) {
    uint32_t seed = 0x00100001;
    for (int index1 = 0; index1 < 0x100; ++index1) {
        for (int i = 0; i < 5; ++i) {
            int index2 = i * 0x100 + index1;
            seed = (seed * 125 + 3) % 0x2AAAAB;
            uint32_t temp1 = (seed & 0xFFFF) << 0x10;
            seed = (seed * 125 + 3) % 0x2AAAAB;
            uint32_t temp2 = (seed & 0xFFFF);
            table[index2] = temp1 | temp2;
        }
    }
}

int runCudaBatch(int zLen, uint64_t startIdx, uint64_t count, uint32_t targetA, uint32_t targetB, FILE* fout) {
    int h_flag = 0;
    cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int));

    if (h_flag) return h_flag;
    char* d_matches;
    int* d_matchCount;
    cudaMalloc(&d_matches, MAX_MATCHES * MAX_FILENAME_LEN);
    cudaMalloc(&d_matchCount, sizeof(int));
    cudaMemset(d_matchCount, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    bruteForceKernel<<<blocks, threadsPerBlock>>>(
            zLen, startIdx, count, targetA, targetB, d_matches, d_matchCount
    );
    cudaDeviceSynchronize();

    int h_matchCount = 0;
    cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);
    h_matchCount = std::min(h_matchCount, MAX_MATCHES);

    char h_matches[MAX_MATCHES][MAX_FILENAME_LEN];
    cudaMemcpy(h_matches, d_matches, h_matchCount * MAX_FILENAME_LEN, cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_matchCount; ++i) {
        fprintf(fout, "%s\n", h_matches[i]);
        fflush(fout);
    }

    cudaFree(d_matches);
    cudaFree(d_matchCount);
    return h_flag;
}



std::string getStartCandidate(std::string path, std::string prefix, std::string suffix) {
    std::string full = path;

    // Remove prefix and suffix
    if (full.rfind(prefix, 0) != 0 || full.size() <= prefix.size() + suffix.size()) {
        fprintf(stderr, "Invalid start filename format.\n");
        return nullptr;
    }

    std::string middle = full.substr(prefix.size(), full.size() - prefix.size() - suffix.size());

    // Remove backslash
    middle.erase(std::remove(middle.begin(), middle.end(), '\\'), middle.end());
    return middle;
}

std::string make_bound_string(std::string input, int zLen) {
    // return a copy of start_filename. Append the last character until the result string has length zLen
    std::string result = input;
    for (int i = input.size(); i < zLen; ++i) {
        result += result.back();
    }
    return result.substr(0, zLen);
}

std::string remove_prefix_and_suffix(std::string base, std::string prefix, std::string suffix) {
    std::string str = base;
    if (str.find(prefix) == 0) {
        str = str.substr(prefix.length());
    }
    if (str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0) {
        str = str.substr(0, str.size() - suffix.size());
    }
    str.erase(std::remove(str.begin(), str.end(), '\\'), str.end());
    return str;
}

std::string getLowerBound(const std::string& input) {
    std::string result = input;
    if (result.empty()) return std::string(16, ' ');

    // Find index of the last character
    char& lastChar = result.back();
    auto pos = alphabet.find(lastChar);
    if (pos == std::string::npos || pos + 1 >= alphabet.size()) {
        fprintf(stderr, "Cannot bump last character or character not in alphabet\n");
    }
    lastChar = alphabet[pos - 1];

    // Pad with underscores to length 16
    result.resize(16, '_');
    return result;
}

std::string getUpperBound(const std::string& input) {
    std::string result = input;
    if (result.empty()) return std::string(16, ' ');

    // Find index of the last character
    char& lastChar = result.back();
    auto pos = alphabet.find(lastChar);
    if (pos == std::string::npos || pos + 1 >= alphabet.size()) {
        fprintf(stderr, "Cannot bump last character or character not in alphabet\n");
    }
    lastChar = alphabet[pos + 1];

    // Pad with spaces to length 16
    result.resize(16, ' ');
    return result;
}


int main(int argc, char* argv[]) {
    if (argc < 9 || (strcmp(argv[1], "continuous") != 0 && strcmp(argv[1], "bounded") != 0)) {
        fprintf(stderr, "Usage: %s <continuous|bounded> <startCandidate> <prefix> <suffix> <lowerBound> <upperBound> <targetHashA> <targetHashB>\n", argv[0]);
        return 1;
    }
    if (alphabet.size() != ALPHABET_SIZE) {
        // This is just a check to make sure we don't change the alphabet without updating its size
        fprintf(stderr, "Alphabet size mismatch. Expected %d, got %zu\n", ALPHABET_SIZE, alphabet.size());
        return 1;
    }

    std::string operation = argv[1];
    std::string prefix = argv[3];
    std::string suffix = argv[4];
    std::string start_candidate = getStartCandidate(argv[2], prefix, suffix);
    std::string lowerBound = argv[5];
    std::string upperBound = argv[6];
    uint32_t target_hash_A = std::stoul(argv[7], nullptr, 16);
    uint32_t target_hash_B = std::stoul(argv[8], nullptr, 16);

    std::string lower = remove_prefix_and_suffix(lowerBound, prefix, suffix);
    std::string upper = remove_prefix_and_suffix(upperBound, prefix, suffix);
    lower = lower.substr(0, lower.size() - suffix.size());

    short prefix_size = prefix.size();
    short suffix_size = suffix.size();
    cudaMemcpyToSymbol(d_prefix_size, &prefix_size, sizeof(prefix_size));
    cudaMemcpyToSymbol(d_suffix_size, &suffix_size, sizeof(suffix_size));
    cudaMemcpyToSymbol(d_prefix, prefix.c_str(), prefix_size + 1);
    cudaMemcpyToSymbol(d_suffix, suffix.c_str(), suffix_size + 1);
    cudaMemcpyToSymbol(d_lowerBound, lowerBound.c_str(), lowerBound.size() + 1);
    cudaMemcpyToSymbol(d_upperBound, upperBound.c_str(), upperBound.size() + 1);
    cudaMemcpyToSymbol(lowerBound, lower.c_str(), lower.size() + 1);
    cudaMemcpyToSymbol(upperBound, upper.c_str(), upper.size() + 1);

    std::string lowerBoundLimit = getLowerBound(lower);
    std::string upperBoundLimit = getUpperBound(upper);

    printf("candidate: '%s'\n", start_candidate.c_str());
    printf("prefix: '%s'\n", prefix.c_str());
    printf("suffix: '%s'\n", suffix.c_str());
    printf("lowerBound: '%s'\n", lowerBound.c_str());
    printf("upperBound: '%s'\n", upperBound.c_str());
    printf("lower: '%s'\n", lower.c_str());
    printf("upper: '%s'\n", upper.c_str());
    printf("lowerBoundLimit: '%s'\n", lowerBoundLimit.c_str());
    printf("upperBoundLimit: '%s'\n", upperBoundLimit.c_str());
    printf("hashA: '%X'\n", target_hash_A);
    printf("hashB: '%X'\n", target_hash_B);

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable));

    FILE* fout = fopen("matches.txt", "a");
    if (!fout) {
        perror("fopen");
        return 1;
    }

    bool found_match = false;
    int zLen = start_candidate.size();
    while (true) {
        std::string start_bound = make_bound_string(start_candidate, zLen);

        uint64_t startIdx = stringToIndex(start_bound);
        uint64_t endIdx = stringToIndex(make_bound_string(upperBoundLimit, zLen));
        printf("Starting at '%s'. Char length = %d → Total combinations: %llu\n", start_bound.c_str(), zLen, (unsigned long long)(endIdx - startIdx));

        const uint64_t batchSize = 1000000;
        for (uint64_t i = startIdx; i < endIdx; i += batchSize) {
            uint64_t count = std::min(batchSize, endIdx - i);
            if (runCudaBatch(zLen, i, count, target_hash_A, target_hash_B, fout) == 1) {
                found_match = true;
                goto breakfree;
            }
        }
        zLen += 1;
        start_candidate = lowerBoundLimit;
        if (operation == "bounded") {
            printf("Reached the upper limit; exiting");
            goto breakfree;
        }
    }
breakfree:

    fclose(fout);
    return found_match ? 0 : 1;
}
