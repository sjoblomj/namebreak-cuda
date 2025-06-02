#include <stdio.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <list>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <utility>


using Entry = std::pair<uint64_t, std::pair<std::string, std::string>>;

class TopNList {
public:
    TopNList(size_t capacity) : max_size(capacity) {}

    void insert(uint64_t score, std::pair<std::string, std::string> strings) {
        Entry new_entry = {score, strings};

        // If list has space, insert and sort
        if (entries.size() < max_size) {
            entries.push_back(new_entry);
            entries.sort(compare_desc);
            return;
        }

        // List is full — compare with the smallest (last in sorted list)
        auto last = std::prev(entries.end());
        if (score > last->first) {
            entries.pop_back();
            entries.push_back(new_entry);
            entries.sort(compare_desc);
        }
    }

    std::list<Entry> getEntries() const {
        return entries;
    }

private:
    static bool compare_desc(const Entry& a, const Entry& b) {
        return a.first > b.first;
    }

    size_t max_size;
    std::list<Entry> entries;
};


const std::string alphabet = " ()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
unsigned long dwCryptTable[0x500];

unsigned long HashString(const char *lpszFileName, unsigned long dwHashType) {
    unsigned char *key = (unsigned char *)lpszFileName;
    unsigned long seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    int ch;

    while (*key != 0) {
        ch = toupper(*key++);

        seed1 = dwCryptTable[(dwHashType << 8) + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

void InitializeCryptTable() {
    unsigned long seed   = 0x00100001;
    unsigned long index1 = 0;
    unsigned long index2 = 0;
    int i;

    for (index1 = 0; index1 < 0x100; index1++) {
        for (index2 = index1, i = 0; i < 5; i++, index2 += 0x100) {
            unsigned long temp1, temp2;

            seed  = (seed * 125 + 3) % 0x2AAAAB;
            temp1 = (seed & 0xFFFF) << 0x10;

            seed  = (seed * 125 + 3) % 0x2AAAAB;
            temp2 = (seed & 0xFFFF);

            dwCryptTable[index2] = (temp1 | temp2);
        }
    }
}

uint64_t stringToIndex(const std::string& str) {
    uint64_t index = 0;
    for (char c : str) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid character");
        }
        index = index * alphabet.size() + pos;
    }
    return index;
}

std::vector<std::pair<uint64_t, std::pair<std::string, std::string>>> computeDistances(
    const std::vector<std::string>& strings
) {
    std::vector<std::pair<uint64_t, std::pair<std::string, std::string>>> distances;
    for (size_t i = 1; i < strings.size(); ++i) {
        uint64_t a = stringToIndex(strings[i - 1]);
        uint64_t b = stringToIndex(strings[i]);
        distances.push_back({b - a, {strings[i - 1], strings[i]}});
    }
    return distances;
}



void perform_analysis(std::map<int, std::vector<std::string>> length_to_string_map) {
    for (auto& entry : length_to_string_map) {
        std::vector<std::string> strings = entry.second;
        std::sort(strings.begin(), strings.end());
        std::vector<std::pair<uint64_t, std::pair<std::string, std::string>>> distances = computeDistances(strings);
        std::sort(distances.begin(), distances.end());

        TopNList topEntries(25);
        uint64_t total = 0;

        for (auto d : distances) {
            total += d.first;
            topEntries.insert(d.first, d.second);
        }
        double avgGap = static_cast<double>(total) / distances.size();
        uint64_t medianGap = distances[distances.size() / 2].first;

        std::cout << "## For strings of length " << entry.first << ":\n";
        std::cout << "Number of strings: "  << strings.size() << "\n\n";

        int i = 0;
        for (auto e : topEntries.getEntries()) {
            std::cout << "### Gap number " << ++i << std::endl;
            std::cout << "- Median gap      : "  << medianGap << "\n";
            std::cout << "- Average gap     : "  << std::fixed << std::setprecision(0) << avgGap << "\n";
            std::cout << "Curr gap          : "  << e.first << "\n";
            std::cout << "Curr gap string 1 : '" << e.second.first  << "'\n";
            std::cout << "Curr gap string 2 : '" << e.second.second << "'\n\n";
        }
        std::cout << std::endl << std::endl;
    }
}

int try_name(std::string name, unsigned long target_hash_A, unsigned long target_hash_B, bool print_match = false, bool print_unmatch = false) {

    unsigned int hashA = (unsigned int) HashString(name.c_str(), 1);
    unsigned int hashB = (unsigned int) HashString(name.c_str(), 2);

    if (hashA == target_hash_A && hashB == target_hash_B) {
        printf("NAME MATCHES BOTH HASHES!!\n%s\n", name.c_str());
        return 0;
    } else if (hashA == target_hash_A || hashB == target_hash_B) {
        if (print_match) {
            printf("Name matches one hash %s\n", name.c_str());
        }
        return 1;
    } else {
        if (print_unmatch) {
            printf("Name is invalid %s\n", name.c_str());
        }
        return -1;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 6 && argc != 4) {
        printf("# Name Checker 1.0 by Johan Sjöblom\n");
        printf("This program computes MPQ hashes to see if the given name\n");
        printf("matches the given hashes, or if the names inside the given\n");
        printf("file matches the given hashes after prepending a prefix\n");
        printf("and appending a suffix.\n\n");
        printf("Usage: %s <name> <hashA> <hashB>\n", argv[0]);
        printf("Usage: %s <filename> <hashA> <hashB> <prefix> <suffix>\n", argv[0]);
        if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
            return 0;
        }
        return 1;
    }
    InitializeCryptTable();

    std::string name = argv[1];
    unsigned long target_hash_A = std::stoul(argv[2], nullptr, 16);
    unsigned long target_hash_B = std::stoul(argv[3], nullptr, 16);
    if (argc == 4) {
        int return_value = 1;
        if (try_name(name, target_hash_A, target_hash_B, true, false) == 0) {
            return_value = 0;
        }
        for (size_t i = 0; i <= name.size() && return_value; ++i) {
            std::string modified = name;
            modified.insert(i, 1, '\\'); // Insert a single backslash at position i
            if (try_name(modified, target_hash_A, target_hash_B, true, false) == 0) {
                return_value = 0;
            }
        }
        if (return_value == 0) {
            printf("Both hashes match! '%s'\n", name.c_str());
        } else {
            printf("Both hashes do not match.\n");
        }
        return return_value;
    }
    std::string prefix = argv[4];
    std::string suffix = argv[5];

    std::ifstream file(name);
    if (!file) {
        std::cerr << "Error: could not open file.\n";
        return 1;
    }


    std::map<int, std::vector<std::string>> length_to_string_map;
    std::string line;
    while (std::getline(file, line)) {
        if (line == "" || line == "\\") {
            continue;
        }

        std::string s = line;
        size_t pos = s.find('\\');
        if (pos != std::string::npos) {
            s.erase(pos, 1);
        }
        length_to_string_map[s.length()].push_back(s);

        std::string name = prefix + line + suffix;
        if (try_name(name, target_hash_A, target_hash_B, false, true) == 0) {
            file.close();
            return 0;
        }
    }
    file.close();
    perform_analysis(length_to_string_map);

    return 1;
}
