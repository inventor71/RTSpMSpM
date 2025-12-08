#include <vector>
#include <algorithm>
#include <tuple>

// ============================================================================
// Helper Functions

/**
 * Load in coo from data file and convert to 0-based indexing with sorting
 */
void cooFromFile( const std::string& filePath, int** rowArr, int** colArr,
                  float** valArr, int* rowSize, int* colSize,
                  uint64_t* arrSize)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filePath << std::endl;
        return;
    }
    std::string line;
    // Skip header lines here
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix metadata (num_rows, num_cols, nnz)
    int rows, cols;
    uint64_t nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    *rowSize = rows;
    *colSize = cols;
    *arrSize = nnz;

    // Read COO data (1-based in file) into vector for sorting
    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(nnz);

    for (uint64_t i = 0; i < nnz; ++i) {
        int row, col;
        float val;
        file >> row >> col >> val;
        // Convert to 0-based indexing
        entries.push_back(std::make_tuple(row - 1, col - 1, val));
    }
    file.close();

    // Sort by row (primary) then column (secondary)
    std::sort(entries.begin(), entries.end());

    // Allocate memory for COO arrays
    *rowArr = new int[*arrSize];
    *colArr = new int[*arrSize];
    *valArr = new float[*arrSize];

    // Fill arrays with sorted, 0-based data
    for (uint64_t i = 0; i < nnz; ++i) {
        (*rowArr)[i] = std::get<0>(entries[i]);
        (*colArr)[i] = std::get<1>(entries[i]);
        (*valArr)[i] = std::get<2>(entries[i]);
    }
}

/**
 * Print coo to output file (convert back to 1-based indexing for MTX format)
 */
void printCooToFile( const std::string& filePath, const int* rowArr,
                     const int* colArr, const float* valArr, const int rowSize,
                     const int colSize, uint64_t arrSize)
{
    // Open File
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    // MTX header
    outFile << "%%MatrixMarket matrix coordinate real general\n";
    // Dimensions and non-zero count
    outFile << rowSize << " " << colSize << " " << arrSize << "\n";

    for (uint64_t i = 0; i < arrSize; ++i){
        // Convert back to 1-based indexing for output
        outFile << (rowArr[i] + 1) << " " << (colArr[i] + 1) << " " << valArr[i] << "\n";
    }
    outFile.close();
}

void coo_to_csr(const int* cooRow, uint64_t nnz, int num_rows, int* csrRowPtr)
{
    // Note: Input is already 0-based and sorted from cooFromFile
    // Build CSR format with 0-based indexing
    std::fill(csrRowPtr, csrRowPtr + num_rows + 1, 0);

    // Count occurrences of each row index
    for (uint64_t i = 0; i < nnz; ++i) {
        csrRowPtr[cooRow[i] + 1]++;
    }

    // Cumulative sum to get row pointers (0-based)
    for (int i = 0; i < num_rows; ++i) {
        csrRowPtr[i + 1] += csrRowPtr[i];
    }
}

void csr_to_coo(const int* csrRowPtr, int num_rows, uint64_t nnz, int* cooRow, 
                bool is_one_based) 
{
    // Iterate over each row in CSR format
    for (uint64_t row = 0; row < num_rows; ++row) {
        uint64_t start = csrRowPtr[row] - (is_one_based ? 1 : 0);
        uint64_t end = csrRowPtr[row + 1] - (is_one_based ? 1 : 0);

        // Assign row index to COO format for each non-zero element in this row
        for (uint64_t i = start; i < end; ++i) {
            cooRow[i] = is_one_based ? row + 1 : row;
        }
    }
}

