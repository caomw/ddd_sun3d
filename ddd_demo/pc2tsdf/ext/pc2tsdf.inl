
//
// inline file for pc2tsdf.h
//

namespace pc2tsdf
{

void PointCloudToTSDF::makeTSDF(const PointCloudf &cloud, float voxelSize, float truncationRadius, TSDF &out)
{
    //std::cout << "Point cloud size: " << cloud.m_points.size() << std::endl;
    
    accel.init(cloud.computeBoundingBox(), truncationRadius);
    for (const vec3f &v : cloud.m_points)
    {
        accel.addPoint(v);
    }

    out.voxelSize = voxelSize;
    out.truncationRadius = truncationRadius;

    const bbox3f bbox = accel.bbox;

    const vec3f buffer = vec3f(out.voxelSize, out.voxelSize, out.voxelSize);
    out.worldOrigin = bbox.getMin() - buffer;
    
    const int dimX = math::ceil(bbox.getExtentX() / out.voxelSize) + 2;
    const int dimY = math::ceil(bbox.getExtentY() / out.voxelSize) + 2;
    const int dimZ = math::ceil(bbox.getExtentZ() / out.voxelSize) + 2;
    out.data.allocate(dimX, dimY, dimZ);

    //std::cout << "Computing TSDF: " << out.data.getDimensions().toString(",") << std::endl;
    for (int z = 0; z < dimZ; z++)
    {
        //if (z % 10 == 0)
        //    std::cout << "z = " << z << std::endl;
        for (int y = 0; y < dimY; y++)
            for (int x = 0; x < dimX; x++)
            {
                out.data(x, y, z) = computeTSDFValue(cloud, out.getVoxelCenter(vec3ui(x, y, z)), out.truncationRadius);
            }
    }
}

float PointCloudToTSDF::computeTSDFValue(const PointCloudf &cloud, const vec3f &pos, float truncationRadius)
{
    const auto result = accel.findClosestPoint(pos);
    const float dist = sqrtf(result.second);
    return std::min(truncationRadius, dist);
}

}