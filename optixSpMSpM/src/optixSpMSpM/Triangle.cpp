#include "Triangle.h"

#include <sutil/sutil.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <iostream>

#include "Util.h"

// #define MAX_NUM_ROW 55548
#define MAX_NUM_ROW 42548

/**
 *
 * Generate triangle from file
 * Sets up the triangle parameter
 * @param fileName, file containing a subset of matrix data needed to be processed
 *        matrix data in COO format (x, y, value)
 *        stored in Triangle as points = (x, y, 0) and value = (value)
 *
 */
Triangle::Triangle( const OptixDeviceContext context, const std::string& fileName, float minval, float maxval)
    : m_context( context )
{
    
    std::ifstream file(fileName);
    std::string line;

    std::ifstream input( fileName.c_str(), std::ios::in );
    SUTIL_ASSERT_MSG( input.is_open(), "Unable to open " + fileName + "." );

    bool isFirstDataLine = true;
    int rows, cols;
    uint64_t nonZeros;

    bool isLargerThanMem = false;
    int midRowStart, midRowEnd, midColStart, midColEnd;

    while (std::getline(file, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '%') {
            continue;
        }

        if (isFirstDataLine) {
            // Read the first data line containing dimensions
            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nonZeros)) {
                std::cerr << "Error reading matrix dimensions." << std::endl;
                return;
            }

            this->m_row = rows;
            this->m_col = cols;
            isFirstDataLine = false;
            continue;
        }

        std::istringstream iss(line);
        float x, y, val;
        if (!(iss >> x >> y >> val)) {
            // Handle parsing error
            continue; 
        }

        float3 vertices[3];

        float ycoord_min = minval * val;
        
        vertices[0] = make_float3( x - 1.5, ycoord_min, minval ); // convert to 0-based
        vertices[1] = make_float3( x - 0.5, ycoord_min, minval );
        vertices[2] = make_float3( x - 1.0, maxval * val, maxval );            

        m_points.push_back(vertices[0]); // Z-coordinate is 0 for 2D data
        m_points.push_back(vertices[1]);
        m_points.push_back(vertices[2]);

        m_value.push_back(y - 1.0); // convert to 0-based
    }
}

Triangle::~Triangle() {}

/**
 *
 * @return a list of float3 data that marks points of the triangle vertices
 *         each data should be in the format of (x, y, 0) which marks 
 *         the x and y axis in the matrix of the current value.
 *
 */
std::vector<float3> Triangle::points() const
{
    return m_points;
}

/**
 *
 * @return the list of value of the matrix data
 * correspond to the (x,y) coordinates in points()
 *
 */
std::vector<float>  Triangle::value() const
{
    return m_value;
}

float Triangle::maxvalue() const
{
    return *std::max_element(m_value.begin(), m_value.end());
}

std::pair<int,int> Triangle::printDim() const
{
    return std::make_pair(m_row, m_col);
}

void Triangle::printPoints() const
{
    for( auto point : m_points )
    {
        std::cout << point << std::endl;
    }
    return;
}