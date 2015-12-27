#ifndef CORE_MESH_POINTCLOUD_H_
#define CORE_MESH_POINTCLOUD_H_

namespace ml {

template <class FloatType>
class PointCloud {
public:
	PointCloud() {}

	PointCloud(const std::vector < vec3<FloatType>>& points) {
		m_points = points;
	}

	PointCloud(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
	}
	void operator=(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
	}

	bool hasNormals() const {
		return m_normals.size() > 0;
	}
	bool hasColors() const {
		return m_colors.size() > 0;
	}

	void clear() {
		m_points.clear();
		m_normals.clear();
		m_colors.clear();
	}

	bool isConsistent() const {
		bool is = true;
		if (m_normals.size() > 0 && m_normals.size() != m_points.size())	is = false;
		if (m_colors.size() > 0 && m_colors.size() != m_points.size())		is = false;
		return is;
	}

	bool isEmpty() const {
		return m_points.size() == 0;
	}

    //! Computes the bounding box of the mesh (not cached!)
    BoundingBox3<FloatType> computeBoundingBox() const {
        BoundingBox3<FloatType> bb;
        for (size_t i = 0; i < m_points.size(); i++) {
            bb.include(m_points[i]);
        }
        return bb;
    }


	std::vector<vec3<FloatType>> m_points;
	std::vector<vec3<FloatType>> m_normals;
	std::vector<vec4<FloatType>> m_colors;
private:
};

typedef PointCloud<float>	PointCloudf;
typedef PointCloud<double>	PointCloudd;

} // namespace ml


#endif