
#ifndef CORE_GRAPHICS_BOUNDINGBOX3_H_
#define CORE_GRAPHICS_BOUNDINGBOX3_H_

#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>

namespace ml {

template<class T>
class OrientedBoundingBox3;

template<class FloatType>
class BoundingBox3
{
public:

	BoundingBox3() {
		reset();
	}

	explicit BoundingBox3(const std::vector< vec3<FloatType> >& verts) {
		reset();
        for (const auto &v : verts)
			include(v);
	}

	explicit BoundingBox3(typename std::vector<vec3<FloatType>>::const_iterator pBegin, typename std::vector<vec3<FloatType>>::const_iterator pEnd) {
		reset();
		for (const auto& iter = pBegin; iter != pEnd; iter++) {
			include(*iter);
		}
	}

	BoundingBox3(const vec3<FloatType>& p0, const vec3<FloatType>& p1, const vec3<FloatType>& p2) {
		reset();
		include(p0);
		include(p1);
		include(p2);
	}

	BoundingBox3(const vec3<FloatType>& minBound, const vec3<FloatType>& maxBound) {
		reset();
		minB = minBound;
		maxB = maxBound;
	}

    BoundingBox3(const BoundingBox3<FloatType> &bbox) {
        minB = bbox.minB;
        maxB = bbox.maxB;
    }

    explicit BoundingBox3(const OrientedBoundingBox3<FloatType> &obb) {
        reset();
		std::vector< vec3 <FloatType > > vertices = obb.getVertices();
        for (const auto &v : vertices)
		//for (const auto &v : obb.getVertices())
            include(v);
    }

	void reset() {
		minX = minY = minZ = std::numeric_limits<FloatType>::max();
		maxX = maxY = maxZ = -std::numeric_limits<FloatType>::max();
	}

	void include(const BoundingBox3 &other) {
		if (other.minX < minX)	minX = other.minX;
		if (other.minY < minY)	minY = other.minY;
		if (other.minZ < minZ)	minZ = other.minZ;

		if (other.maxX > maxX)	maxX = other.maxX;
		if (other.maxY > maxY)	maxY = other.maxY;
		if (other.maxZ > maxZ)	maxZ = other.maxZ;
	}

	void include(const vec3<FloatType> &v) {
		if (v.x < minX)	minX = v.x;
		if (v.y < minY)	minY = v.y;
		if (v.z < minZ)	minZ = v.z;

		if (v.x > maxX)	maxX = v.x;
		if (v.y > maxY)	maxY = v.y;
		if (v.z > maxZ)	maxZ = v.z;
	}

    void include(const std::vector<vec3<FloatType>> &v) {
        for (const auto &p : v)
            include(p);
    }

	bool isInitialized() const {
		return (minX != std::numeric_limits<FloatType>::max());
	}
	bool isValid() const {
		return (minX <= maxX && minY <= maxY && minZ <= maxZ);
	}

	void getVertices(vec3<FloatType> *result) const {
		result[0] = vec3<FloatType>(minX, minY, minZ);
		result[1] = vec3<FloatType>(maxX, minY, minZ);
		result[2] = vec3<FloatType>(maxX, maxY, minZ);
		result[3] = vec3<FloatType>(minX, maxY, minZ);
		result[4] = vec3<FloatType>(minX, minY, maxZ);
		result[5] = vec3<FloatType>(maxX, minY, maxZ);
		result[6] = vec3<FloatType>(maxX, maxY, maxZ);
		result[7] = vec3<FloatType>(minX, maxY, maxZ);
	}

	std::vector< vec3<FloatType> > getVertices() const {
		std::vector< vec3<FloatType> > result;
		result.resize(8);

		getVertices(result.data());

		return result;
	}





	//! point collision
	bool intersects(const vec3<FloatType>& p) const {
        if (p.x >= minX && p.x <= maxX &&
            p.y >= minY && p.y <= maxY &&
            p.z >= minZ && p.z <= maxZ)
            return true;
		return false;
	}

	//! triangle collision
	// bool intersects(const vec3<FloatType>& p0, const vec3<FloatType>& p1, const vec3<FloatType>& p2) const {
	// 	return intersection::intersectTriangleAABB(minB, maxB, p0, p1, p2);
	// }

	//! bounding box collision
	bool intersects(const BoundingBox3<FloatType>& other) const {
		return 
			minX <= other.maxX && other.minX <= maxX &&
			minY <= other.maxY && other.minY <= maxY &&
			minZ <= other.maxZ && other.minZ <= maxZ;
	}

    //! transformed bounding box collision
    //bool collision(const BoundingBox3<FloatType>& other, const matrix4x4<FloatType> &transform) const {
    //    BoundingBox3<FloatType> otherTransformed = other * transform;
    //    return collision(otherTransformed);
    //}

	FloatType getMaxExtent() const {
		FloatType d0 = maxX - minX;
		FloatType d1 = maxY - minY;
		FloatType d2 = maxZ - minZ;
		return math::max(d0, d1, d2);
	}

	FloatType getExtentX() const {
		return maxX - minX;
	}

	FloatType getExtentY() const {
		return maxY - minY;
	}

	FloatType getExtentZ() const {
		return maxZ - minZ;
	}

	vec3<FloatType> getExtent() const {
		return vec3<FloatType>(maxX - minX, maxY - minY, maxZ - minZ);
	}

	vec3<FloatType> getMin() const {
		return vec3<FloatType>(minX, minY, minZ);
	}

	vec3<FloatType> getMax() const {
		return vec3<FloatType>(maxX, maxY, maxZ);
	}

	vec3<FloatType> getCenter() const {
		vec3<FloatType> center = getMin() + getMax();
		center *= (FloatType)0.5;
		return center;
	}

	void setMin(const vec3<FloatType>& minValue) {
		minX = minValue.x;
		minY = minValue.y;
		minZ = minValue.z;
	}

	void setMax(const vec3<FloatType>& maxValue) {
		maxX = maxValue.x;
		maxY = maxValue.y;
		maxZ = maxValue.z;
	}

	void setMinX(FloatType v) { minX = v; }
	void setMinY(FloatType v) { minY = v; }
	void setMinZ(FloatType v) { minZ = v; }
	void setMaxX(FloatType v) { maxX = v; }
	void setMaxY(FloatType v) { maxY = v; }
	void setMaxZ(FloatType v) { maxZ = v; }

	FloatType getMinX() const { return minX; }
	FloatType getMinY() const { return minY; }
	FloatType getMinZ() const { return minZ; }
	FloatType getMaxX() const { return maxX; }
	FloatType getMaxY() const { return maxY; }
	FloatType getMaxZ() const { return maxZ; }

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType x, FloatType y, FloatType z) {

		FloatType scale[] = {x, y, z};
		for (unsigned int i = 0; i < 3; i++) {
			FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+3]);
			FloatType diff = parameters[i+3] - parameters[i];
			diff *= scale[i];
			diff *= (FloatType)0.5;
			parameters[i] = center - diff;
			parameters[i+3] = center + diff;
		}
	}

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType t) {
		for (unsigned int i = 0; i < 3; i++) {
			FloatType center = (FloatType)0.5 * (parameters[i] + parameters[i+3]);
			FloatType diff = parameters[i+3] - parameters[i];
			diff *= t;
			diff *= (FloatType)0.5;
			parameters[i] = center - diff;
			parameters[i+3] = center + diff;
		}
	}

	void translate(const vec3<FloatType>& t) {
		minB += t;
		maxB += t;
	}

	//! scales the bounding box (see scale)
	BoundingBox3<FloatType> operator*(FloatType t) const {
		BoundingBox3<FloatType> res = *this;
		res.scale(t);
		return res;
	}

	void setUnitCube() {
		minX = minY = minZ = 0;
		maxX = maxY = maxZ = 1;
	}

    void operator = (const BoundingBox3<FloatType> &bbox)
    {
        minB = bbox.minB;
        maxB = bbox.maxB;
    }

protected:

#ifdef _WIN32
  // boost archive serialization functions
  friend class boost::serialization::access;
  template <class Archive>
  inline void serialize(Archive& ar, const unsigned int verion) {
    ar & boost::serialization::make_array(parameters, 6);
  }
#endif

	union {
		struct {
			vec3<FloatType> minB;
			vec3<FloatType> maxB;
		};
		struct {
			FloatType minX, minY, minZ;
			FloatType maxX, maxY, maxZ;
		};
		FloatType parameters[6];
	};
};

template<class FloatType>
std::ostream& operator<< (std::ostream& s, const BoundingBox3<FloatType>& bb) {
	s << bb.getMin() << std::endl << bb.getMax() << std::endl;
	return s;
}

typedef BoundingBox3<float> BoundingBox3f;
typedef BoundingBox3<double> BoundingBox3d;

typedef BoundingBox3<float> bbox3f;
typedef BoundingBox3<double> bbox3d;

}  // namespace ml



#endif  // CORE_GRAPHICS_BOUNDINGBOX3_H_
