#include <iostream>
#include <optional>

int main() {
    std::optional<int> opt = 42;
    if (opt) {
        std::cout << "C++17 is supported! Value: " << *opt << std::endl;
    } else {
        std::cout << "C++17 is not supported!" << std::endl;
    }
    return 0;
}
