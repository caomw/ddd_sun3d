
#ifndef __PC2TSDF_H_
#define __PC2TSDF_H_

#include "ext/common.h"
#include "ext/utility.h"
#include "ext/stringUtil.h"
#include "ext/vec3.h"
#include "ext/vec4.h"
#include "ext/mat4.h"
#include "ext/grid3.h"
#include "ext/boundingBox3.h"

#include "ext/plyHeader.h"
#include "ext/pointCloud.h"
#include "ext/pointCloudIO.h"

using namespace ml;

namespace pc2tsdf
{

struct TSDFHeader
{
    int headerSize;

    int dimX;
    int dimY;
    int dimZ;

    vec3f worldOrigin;
    float voxelSize;
    float truncationRadius;
};

struct TSDF
{
    vec3f getVoxelCenter(const vec3ui &cell) const
    {
        return worldOrigin + vec3f(cell) * voxelSize + vec3f(voxelSize * 0.5f);
    }

    void saveBinary(const std::string &filename) const
    {
        FILE *file = util::checkedFOpen(filename.c_str(), "wb");
        if (!file)
        {
            std::cout << "Failed to open file: " << file << std::endl;
            return;
        }

        TSDFHeader header;
        header.headerSize = sizeof(TSDFHeader);
        header.dimX = (int)data.getDimensions().x;
        header.dimY = (int)data.getDimensions().y;
        header.dimZ = (int)data.getDimensions().z;

        header.worldOrigin = worldOrigin;
        header.voxelSize = voxelSize;
        header.truncationRadius = truncationRadius;

        util::checkedFWrite(&header, sizeof(header), 1, file);

        util::checkedFWrite(data.getData(), sizeof(float), header.dimX * header.dimY * header.dimZ, file);

        fclose(file);
    }

    vec3f worldOrigin;
    float voxelSize;
    float truncationRadius;

    Grid3f data;
};

struct UniformAccelerator
{
    struct Entry
    {
        std::vector<vec3f> points;
    };

    bbox3f bbox;
    float cubeSize;
    Grid3<Entry> data;

    UniformAccelerator() {}
    UniformAccelerator(const std::vector<vec3f> &points, float _cubeSize)
    {
        bbox3f _bbox;
        for (auto &v : points)
            _bbox.include(v);
        init(_bbox, _cubeSize);
        for (auto &v : points)
            addPoint(v);
    }

    void init(const bbox3f &_bbox, float _cubeSize)
    {
        bbox = _bbox;
        cubeSize = _cubeSize;
        int dimX = math::ceil(bbox.getExtentX() / cubeSize) + 1;
        int dimY = math::ceil(bbox.getExtentY() / cubeSize) + 1;
        int dimZ = math::ceil(bbox.getExtentZ() / cubeSize) + 1;
        data.allocate(dimX, dimY, dimZ);
    }

    vec3i getCoord(const vec3f &pos) const
    {
        int x = math::clamp((int)math::linearMap(bbox.getMinX(), bbox.getMaxX(), 0.0f, (float)data.getDimX(), pos.x), 0, (int)data.getDimX() - 1);
        int y = math::clamp((int)math::linearMap(bbox.getMinY(), bbox.getMaxY(), 0.0f, (float)data.getDimY(), pos.y), 0, (int)data.getDimY() - 1);
        int z = math::clamp((int)math::linearMap(bbox.getMinZ(), bbox.getMaxZ(), 0.0f, (float)data.getDimZ(), pos.z), 0, (int)data.getDimZ() - 1);
        return vec3i(x, y, z);
    }

    void addPoint(const vec3f &pos)
    {
        const vec3i coord = getCoord(pos);
        data(coord.x, coord.y, coord.z).points.push_back(pos);
    }

    //! returns the closest point and the distance squared. If no point is found in the adjacent boxes, returns max-float.
    std::pair<vec3f, float> findClosestPoint(const vec3f &pos) const
    {
        const vec3i baseCoord = getCoord(pos);

        float bestDistSq = std::numeric_limits<float>::max();
        vec3f bestPoint = vec3f(bestDistSq, bestDistSq, bestDistSq);

        for (int xOffset = -1; xOffset <= 1; xOffset++)
            for (int yOffset = -1; yOffset <= 1; yOffset++)
                for (int zOffset = -1; zOffset <= 1; zOffset++)
                {
                    const vec3i coord = baseCoord + vec3i(xOffset, yOffset, zOffset);
                    if (data.isValidCoordinate(coord.x, coord.y, coord.z))
                    {
                        for (auto &p : data(coord.x, coord.y, coord.z).points)
                        {
                            const float distSq = vec3f::distSq(pos, p);
                            if (distSq < bestDistSq)
                            {
                                bestDistSq = distSq;
                                bestPoint = p;
                            }
                        }
                    }
                }

        return std::make_pair(bestPoint, bestDistSq);
    }
};

class PointCloudToTSDF
{
public:
    void makeTSDF(const PointCloudf &cloud, float voxelSize, float truncationRadius, TSDF &out);

private:
    float computeTSDFValue(const PointCloudf &cloud, const vec3f &pos, float truncationRadius);
    
    UniformAccelerator accel;
};

inline void makeTSDF(const PointCloudf &cloud, float voxelSize, float truncationRadius, TSDF &out)
{
    PointCloudToTSDF maker;
    maker.makeTSDF(cloud, voxelSize, truncationRadius, out);
}

}

#include "ext/pc2tsdf.inl"

#endif