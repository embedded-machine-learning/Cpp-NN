#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/helpers/human_readable_types.hpp"
#include "../include/helpers/print.hpp"

int main() {
    // Test the human-readable type names
    std::cout << "Human-readable type for int: " << human_readable_type<int> << std::endl;
    std::cout << "Human-readable type for float: " << human_readable_type<float> << std::endl;
    std::cout << "Human-readable type for double: " << human_readable_type<double> << std::endl;
    std::cout << "Human-readable type for void: " << human_readable_type<void> << std::endl;

    std::cout << "=========================================================" << std::endl;
    std::cout << "Human-readable type for Matrix<int, 'BC', 3, 4>: \n\t" << human_readable_type<Matrix<int, "BC", 3, 4> &> << std::endl;
    std::cout << "Human-readable type for PermutedMatrix<'CB', Matrix<int, 'BC', 3, 4>>: \n\t" << human_readable_type<PermutedMatrix<"CB", Matrix<int, "BC", 3, 4>>> << std::endl;
    std::cout << "Human-readable type for Materialized (PermutedMatrix<'CB', Matrix<int, 'BC', 3, 4>>): \n\t"
              << human_readable_type<MaterializedMatrix<PermutedMatrix<"CB", Matrix<int, "BC", 3, 4>>>> << std::endl;
    std::cout << "Human-readable type for DimensionOrder: \n\t" << human_readable_type<DimensionOrder> << std::endl;
    std::cout << "Human-readable type for std::tuple<int, float, double>: \n\t" << human_readable_type<std::tuple<int, float, double>> << std::endl;
    // Add more tests as needed
    return 0;
}
