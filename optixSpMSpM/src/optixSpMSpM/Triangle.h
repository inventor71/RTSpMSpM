#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Aabb.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ostream>

// forwrad declarations
class Context;
class HairProgramGroups;


class Triangle {
    public:
        Triangle( const OptixDeviceContext context, const std::string& fileName, float minval, float maxval );
        virtual ~Triangle();

        std::vector<float3> points() const;
        std::vector<float>  value() const;
        float              maxvalue() const;
        std::pair<int,int> printDim() const;
        void           printPoints() const;
        /**
        virtual void gatherProgramGroups( HairProgramGroups* pProgramGroups ) const;

        std::string programName() const;
        std::string programSuffix() const;*/

        sutil::Aabb  aabb() const { return m_aabb; }

    protected:
        OptixTraversableHandle gas() const;

        void makeOptix() const;
        void clearOptix();

    private:
        //TODO: FileHeader          m_header;
        std::vector<float3> m_points;
        std::vector<float>  m_value;
        int m_row;
        int m_col;

        mutable sutil::Aabb m_aabb;

        OptixDeviceContext m_context = 0;

        friend std::ostream& operator<<( std::ostream& o, const Triangle& triangle );
};

// Output operator for Triangle
std::ostream& operator<<( std::ostream& o, const Triangle& triangle );