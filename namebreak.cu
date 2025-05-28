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

//__device__ __constant__ char d_prefix[] = "PORTRAIT\\U";
__device__ __constant__ char d_prefix[] = "PORTRAIT\\T";
__device__ __constant__ char d_suffix[] = "FID00.SMK";
//const std::string prefix = "PORTRAIT\\U";
const std::string prefix = "PORTRAIT\\T";
const std::string suffix = "FID00.SMK";

/*
// <reference 1>
__device__ __constant__ char d_lowerBound[] = "PORTRAIT\\UDTEMPLAR\\UDTTLK02.SMK";
__device__ __constant__ char d_upperBound[] = "PORTRAIT\\UFENDRAG\\UFDFID00.SMK";
__device__ __constant__ char lowerBound[] = "DTEMPLARUDT";
__device__ __constant__ char upperBound[] = "FENDRAGUFD";
const std::string upperBoundLimit = "FENDRAGUFE      ";
const uint32_t target_hash_A = 0xD962B57C; // UDUKE\UDUFID00.SMK
const uint32_t target_hash_B = 0xC990B138; // UDUKE\UDUFID00.SMK
// </reference 1>
*/
/*
// <reference 2>
__device__ __constant__ char d_lowerBound[] = "PORTRAIT\\UFLAG1\\UF1TLK00.SMK";
__device__ __constant__ char d_upperBound[] = "PORTRAIT\\UFLAG3\\UF3FID00.SMK";
__device__ __constant__ char lowerBound[] = "FLAG1UF1";
__device__ __constant__ char upperBound[] = "FLAG3UF3";
const std::string upperBoundLimit = "FLAG3UF4        ";
const uint32_t target_hash_A = 0x17D0F420; // UFLAG2\UF2FID00.SMK
const uint32_t target_hash_B = 0xA42467DA; // UFLAG2\UF2FID00.SMK
// </reference 2>
*/
// <reference 3>
__device__ __constant__ char d_lowerBound[] = "PORTRAIT\\TTANK\\TTATLK02.SMK";
__device__ __constant__ char d_upperBound[] = "PORTRAIT\\TVULTURE\\TVUFID00.SMK";
__device__ __constant__ char lowerBound[] = "TANKTTA";
__device__ __constant__ char upperBound[] = "VULTURETVU";
const std::string upperBoundLimit = "VULTURETVV     ";
const uint32_t target_hash_A = 0xAEB771C4; // TVESSEL\TVEFID00.SMK
const uint32_t target_hash_B = 0x1162462A; // TVESSEL\TVEFID00.SMK
// </reference 3>

//const uint32_t target_hash_A = 0xEAE4D0CB; // UFENDRAG\UFDFID00.SMK
//const uint32_t target_hash_B = 0x28E9A64B; // UFENDRAG\UFDFID00.SMK

//$FIRSTWORD = "KHADNCR             ";
//$LASTWORD  = "MENGSKUME___________";


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

uint64_t zStringToIndex(const std::string& z) {
    uint64_t index = 0;
    for (char c : z) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid character in Z: '%c'\n", c);
            exit(1);
        }
        index = index * alphabet.size() + pos;
    }
    return index;
}

__device__ void buildCandidate(const char* Z, int zLen, int split, char* out) {
    int i = 0;
    memcpy(out + i, d_prefix, sizeof(d_prefix) - 1);
    i += sizeof(d_prefix) - 1;

    memcpy(out + i, Z, split);
    i += split;

    out[i++] = '\\';

    memcpy(out + i, Z + split, zLen - split);
    i += zLen - split;

    memcpy(out + i, d_suffix, sizeof(d_suffix) - 1);
    i += sizeof(d_suffix) - 1;

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
//    if (!inBounds(Z)) {
//        printf("Outside bounds: '%s'\n", Z);
//        return;
//    }

    for (int split = 1; split < zLen; ++split) {
        buildCandidate(Z, zLen, split, candidate);

        //if (inBounds(candidate)) {
            uint32_t hashA = mpqHash(candidate);
            if (hashA == targetA) {
                printf("Hash A matches: %s\n", candidate);

                uint32_t hashB = mpqHashSeed2(candidate);
                if (hashB == targetB) {
                    // DONE
                    printf("BOTH HASHES MATCH: %s\n", candidate);
                    d_foundMatchFlag = 1;
                }
                int slot = atomicAdd(d_matchCount, 1);
                if (slot < MAX_MATCHES) {
                    memcpy(&d_matches[slot * MAX_FILENAME_LEN], candidate, MAX_FILENAME_LEN);
                }
            }
        //}
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



std::string getFilename(const char* path) {
    std::string full = path;

    // Remove prefix and suffix
    if (full.rfind(prefix, 0) != 0 || full.size() <= prefix.size() + suffix.size()) {
        fprintf(stderr, "Invalid start filename format.\n");
        return nullptr;
    }

    std::string middle = full.substr(prefix.size(), full.size() - prefix.size() - suffix.size());

    // Remove backslash between X and Y
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <starting_filename>\n", argv[0]);
        return 1;
    }
//    printf("Starting index: %llu for filename '%s'\n", (unsigned long long)startIdx, start_filename.c_str());

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable));

    FILE* fout = fopen("matches.txt", "w");
    if (!fout) {
        perror("fopen");
        return 1;
    }

    std::string start_filename = getFilename(argv[1]);
    int zLen = start_filename.size();
    while (true) {
        std::string start_bound = make_bound_string(start_filename, zLen);
//        printf("Starting bound: '%s'\n", start_bound.c_str());

//        uint64_t total = 1;
//        for (int i = 0; i < zLen; ++i) total *= ALPHABET_SIZE;

        uint64_t startIdx = zStringToIndex(start_bound);
//        printf("Char length = %d → Total combinations: %llu\n", zLen, (unsigned long long)total);
        uint64_t endIdx = zStringToIndex(make_bound_string(upperBoundLimit, zLen));
        printf("Starting at '%s'. Char length = %d → Total combinations: %llu\n", start_bound.c_str(), zLen, (unsigned long long)(endIdx - startIdx));
//        printf("Total   : %llu\n", (unsigned long long)total);
//        total -= endIdx;
//        printf("endIdx  : %llu\n", (unsigned long long)endIdx);
//        printf("Total - : %llu\n", (unsigned long long)total);
//        printf("startIdx: %llu\n", (unsigned long long)startIdx);

//        if (zLen > 11) goto breakfree;
        const uint64_t batchSize = 1000000;
        for (uint64_t i = startIdx; i < endIdx; i += batchSize) {
            uint64_t count = std::min(batchSize, endIdx - i);
            if (runCudaBatch(zLen, i, count, target_hash_A, target_hash_B, fout) == 1) goto breakfree;
        }
        zLen += 1;
    }
breakfree:

    fclose(fout);
    return 0;
}
