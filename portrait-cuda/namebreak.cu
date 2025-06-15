#include <cuda_runtime.h>
#include "cpu-utils.h"
#include "constants.h"

// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix

__device__ __constant__ char d_alphabet[ALPHABET_SIZE + 1] = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";

__device__ volatile int d_foundMatchFlag = 0;

__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ char d_lowerBound[64];
__device__ __constant__ char d_upperBound[64];
__device__ __constant__ char lowerBound[64];
__device__ __constant__ char upperBound[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;
__device__ __constant__ uint32_t d_seed1_start;
__device__ __constant__ uint32_t d_seed2_start;

__device__ __constant__ uint32_t d_cryptTable[0x500];

__device__ uint32_t mpqHashWithPrefixCache(
    const char* str,
    int str_len,
    uint32_t* seed1_out,
    uint32_t* seed2_out,
    int reuseLen
) {
    uint32_t seed1, seed2;
    char ch;
    if (reuseLen > 0) {
        seed1 = seed1_out[reuseLen - 1];
        seed2 = seed2_out[reuseLen - 1];
    } else {
        seed1 = d_seed1_start;
        seed2 = d_seed2_start;
    }

    for (int i = reuseLen; i < str_len; ++i) {
        char ch = str[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
        seed1_out[i] = seed1;
        seed2_out[i] = seed2;
    }

    for (int i = 0; i < d_suffix_size; ++i) {
        ch = d_suffix[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

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

__device__ void indexToCandidate(uint64_t index, int candidateLen, char* outCandidate) {
    for (int i = candidateLen - 1; i >= 0; --i) {
        outCandidate[i] = d_alphabet[index % ALPHABET_SIZE];
        index /= ALPHABET_SIZE;
    }
    outCandidate[candidateLen] = '\0';
}

__device__ void buildFilenameWithPrefix(const char* candidate, int candidateLen, int split, char* out) {
    memcpy(out, d_prefix, d_prefix_size);
    short i = d_prefix_size;

    memcpy(out + i, candidate, split);
    i += split;

    out[i++] = '\\';

    memcpy(out + i, candidate + split, candidateLen - split);
    i += candidateLen - split;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

// TODO: Remove
__device__ void buildFilenameWithoutPrefix(const char* candidate, int candidateLen, int split, char* out) {
    memcpy(out, candidate, split);
    out[split] = '\\';
    memcpy(out + split + 1, candidate + split, candidateLen - split);

//    memcpy(out + i, d_suffix, d_suffix_size);
//    i += d_suffix_size;

    out[candidateLen + 1] = '\0';
}

__device__ void buildFilenameWithPrefixWithoutBackslash(const char* candidate, int candidateLen, char* out) {
    short i = 0;
    memcpy(out, d_prefix, d_prefix_size);
    i += d_prefix_size;

    memcpy(out + i, candidate, candidateLen);
    i += candidateLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

// TODO: Remove
__device__ void buildFilenameWithoutPrefixAndBackslash(const char* candidate, int candidateLen, char* out) {
    short i = 0;
    memcpy(out, candidate, candidateLen);
    i += candidateLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

// TODO: Remove
__device__ int my_strcmp (const char * s1, const char * s2) {
    for(; *s1 == *s2; ++s1, ++s2)
        if(*s1 == 0)
            return 0;
    return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}

// TODO: Remove
__device__ bool inBounds(const char* candidate) {
    return my_strcmp(candidate, lowerBound) >= 0 &&
           my_strcmp(candidate, upperBound) <= 0;
}

__global__ void bruteForceKernel(
    int candidateLen,
    uint64_t startIdx,
    int batchPerThread,
    uint64_t totalCount,
    uint32_t targetA,
    uint32_t targetB,
    char* d_matches,
    int* d_matchCount
) {
    uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t threadStart = startIdx + threadId * batchPerThread;
    //if (threadId < 1)
    //    return;

    char candidate[MAX_CANDIDATE_LEN];
    char filename[MAX_FILENAME_LEN];

    char   prevCandidate[10][11];// TODO: Hard coded number
    uint32_t seed1_cache[10][11];// TODO: Hard coded number
    uint32_t seed2_cache[10][11];// TODO: Hard coded number
    for (int i = 0; i <= candidateLen; ++i) {
        //prevCandidate[i][0] = '\0'; // Only initialize the first element to be null; The rest are uninitialized.
        for (int j = 0; j <= candidateLen + 1; j++) {
            prevCandidate[i][j] = '\0';
        }
    }
    for (int i = 0; i < MAX_CANDIDATE_LEN; ++i) {
        candidate[i] = '\0';
    }

    for (int i = 0; i < batchPerThread; ++i) {
        uint64_t idx = threadStart + i;
        if (idx >= startIdx + totalCount) {
            return;
        }
        indexToCandidate(idx, candidateLen, candidate);

        // First try without backslash
        ///

        int reuseLen = 0;
        while (reuseLen < candidateLen && candidate[reuseLen] == prevCandidate[0][reuseLen]) {
            ++reuseLen;
        }
        //buildFilenameWithoutPrefixAndBackslash(candidate, candidateLen, filename);
        //candidate[candidateLen] = '\0';
        uint32_t hashA = mpqHashWithPrefixCache(candidate, candidateLen, seed1_cache[0], seed2_cache[0], reuseLen);

        /*
        //<remove>
        // For verifying correctness
        char filename2[MAX_FILENAME_LEN];
        buildFilenameWithPrefixWithoutBackslash(candidate, candidateLen, filename2);
//        printf("%4i Apa bepa '%s' %i %i\n", i, filename2, reuseLen, candidateLen);
        uint32_t hashAA = mpqHash(filename2);
        if (hashA != hashAA) {
            printf("%i A Hashes do not match! '%s' '%s' %u %u\n", 0, candidate, filename2, hashA, hashAA);
        }
        //</remove>
        */

        if (hashA == targetA) {
            printf("Hash A matches!\n");
            buildFilenameWithPrefixWithoutBackslash(candidate, candidateLen, filename);
            printf("Hash A matches: %s\n", filename);

            uint32_t hashB = mpqHashSeed2(filename);
            if (hashB == targetB) {
                printf("BOTH HASHES MATCH: %s\n", filename);
                d_foundMatchFlag = 1;
            }
            int slot = atomicAdd(d_matchCount, 1);
            if (slot < MAX_MATCHES) {
                memcpy(&d_matches[slot * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
            }
        }
        ///

        // Then try with all possible splits
        for (int split = 1; split <= candidateLen; ++split) {
            buildFilenameWithoutPrefix(candidate, candidateLen, split, filename);

            hashA = mpqHashWithPrefixCache(filename, candidateLen + 1, seed1_cache[split], seed2_cache[split], reuseLen);

            /*
            //<remove>
            // For verifying correctness
            buildFilenameWithPrefix(candidate, candidateLen, split, filename2);
            hashAA = mpqHash(filename2);
            if (hashA != hashAA) {
                printf("%i A Hashes do not match! '%s' '%s' %u %u\n", split, filename, filename2, hashA, hashAA);
            }
            //</remove>
            */

            if (hashA == targetA) {
                printf("Hash A matches!\n");
                buildFilenameWithPrefix(candidate, candidateLen, split, filename);
                printf("Hash A matches: %s\n", filename);

                uint32_t hashB = mpqHashSeed2(filename);
                if (hashB == targetB) {
                    printf("BOTH HASHES MATCH: %s\n", filename);
                    d_foundMatchFlag = 1;
                }

                int slot = atomicAdd(d_matchCount, 1);
                if (slot < MAX_MATCHES) {
                    memcpy(&d_matches[slot * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
                }
            }
        }

        for (int j = reuseLen; j <= candidateLen; ++j) {
            prevCandidate[0][j] = candidate[j];
            for (int split = 1; split <= candidateLen; ++split) {
                if (j < split) {
                    prevCandidate[split][j] = candidate[j];
                } else {
                    prevCandidate[split][j + 1] = candidate[j];
                }
            }
        }
        prevCandidate[0][candidateLen] = '\0';
        for (int split = 1; split <= candidateLen; ++split) {
            prevCandidate[split][candidateLen + 1] = '\0';
        }
    }

//    if (threadId == 0) {
        //printf("Done %lu %i '%s' %X\n", threadStart, batchPerThread, candidate, targetA);
//    }
}


int runCudaBatch(int candidateLen, uint64_t startIdx, uint64_t count, uint32_t targetA, uint32_t targetB, FILE* fout) {
    int h_flag = 0;
    cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int));

    if (h_flag) return h_flag;
    char* d_matches;
    int* d_matchCount;
    cudaMalloc(&d_matches, MAX_MATCHES * MAX_FILENAME_LEN);
    cudaMalloc(&d_matchCount, sizeof(int));
    cudaMemset(d_matchCount, 0, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);




    int batchPerThread = ALPHABET_SIZE * 32;
    int totalThreads = (count + batchPerThread - 1) / batchPerThread;
    int threadsPerBlock = 256;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
//    printf("count: %lu, batchPerThread: %i, totalThreads: %i, blocks: %i, startIdx: %lu\n",
//           count, batchPerThread, totalThreads, blocks, startIdx);

    // TODO: Understand why the candidates end with the same character but "jumps",
    // i.e. goes from R -> Z -> ( -> 2

    bruteForceKernel<<<blocks, threadsPerBlock>>>(
            candidateLen, startIdx, batchPerThread, count, targetA, targetB, d_matches, d_matchCount
    );
    //cudaDeviceSynchronize();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);


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

    std::string lowerBoundLimit = getLowerBound(lower, alphabet);
    std::string upperBoundLimit = getUpperBound(upper, alphabet);

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

    std::pair<uint32_t, uint32_t> pair = mpqHashWithPrefixCache_CPU(prefix.c_str(), h_cryptTable);
    uint32_t seed1_start = pair.first;
    uint32_t seed2_start = pair.second;
    cudaMemcpyToSymbol(d_seed1_start, &seed1_start, sizeof(seed1_start));
    cudaMemcpyToSymbol(d_seed2_start, &seed2_start, sizeof(seed2_start));

    FILE* fout = fopen("matches.txt", "a");
    if (!fout) {
        perror("fopen");
        return 1;
    }

    bool found_match = false;
    int candidateLen = start_candidate.size();
    while (true) {
        std::string start_bound = make_bound_string(start_candidate, candidateLen);

        uint64_t startIdx = stringToIndex(start_bound, alphabet);
        uint64_t endIdx = stringToIndex(make_bound_string(upperBoundLimit, candidateLen), alphabet);
        printf("Starting at '%s'. Char length = %d → Total combinations: %llu\n", start_bound.c_str(), candidateLen, (unsigned long long)(endIdx - startIdx));

        const uint64_t batchSize = ALPHABET_SIZE * 20000;
        for (uint64_t i = startIdx; i < endIdx; i += batchSize) {
            uint64_t count = std::min(batchSize, endIdx - i);
            if (runCudaBatch(candidateLen, i, count, target_hash_A, target_hash_B, fout) == 1) {
                found_match = true;
                goto breakfree;
            }
        }
        candidateLen += 1;
        start_candidate = lowerBoundLimit;
        if (operation == "bounded") {
            printf("Reached the upper limit; exiting");
            goto breakfree;
        }
    }
breakfree:

    fclose(fout);
    return found_match ? 0 : 2;
}
