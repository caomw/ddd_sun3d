
#ifndef CORE_MATH_MATRIX4X4_H_
#define CORE_MATH_MATRIX4X4_H_

namespace ml {

//! This class provides functions to handle 4-dimensional matrices
/*! The arrangement of the matrix is row-like.
    The index of a specific position is:
    <pre>
       0  1   2  3
       4  5   6  7
       8  9  10  11
       12 13 14  15
    </pre>
*/
template <class FloatType> class Matrix4x4
{
public:
	//! An uninitialized matrix
	Matrix4x4() {}

	//! Initialize with values stored in an array
	Matrix4x4(const FloatType* values) {
		for (unsigned int i = 0; i < 16; i++) {
			matrix[i] = values[i];
		}
	}

	//! Initializes the matrix row wise (given 3 row vectors); that last row and column are initialized with 0,0,0,1
	Matrix4x4(const vec3<FloatType> &v0, const vec3<FloatType> &v1, const vec3<FloatType> &v2) {
		matrix2[0][0] = v0.x;	matrix2[0][1] = v0.y;	matrix2[0][2] = v0.z;	matrix2[0][3] = 0.0;
		matrix2[1][0] = v1.x;	matrix2[1][1] = v1.y;	matrix2[1][2] = v1.z;	matrix2[1][3] = 0.0;
		matrix2[2][0] = v2.x;	matrix2[2][1] = v2.y;	matrix2[2][2] = v2.z;	matrix2[2][3] = 0.0;
		matrix2[3][0] = 0.0;	matrix2[3][1] = 0.0;	matrix2[3][2] = 0.0;	matrix2[3][3] = 1.0;
	}

	////! Initializes the matrix row wise (given 3 row vectors); that last row is initialized with 0,0,0,1
	//Matrix4x4(const vec3<FloatType> &v0, const vec3<FloatType> &v1, const vec3<FloatType> &v2, const vec3<FloatType> &v3) {
	//	matrix2[0][0] = v0.x;	matrix2[0][1] = v0.y;	matrix2[0][2] = v0.z;	matrix2[0][3] = v0.w;
	//	matrix2[1][0] = v1.x;	matrix2[1][1] = v1.y;	matrix2[1][2] = v1.z;	matrix2[1][3] = v1.w;
	//	matrix2[2][0] = v2.x;	matrix2[2][1] = v2.y;	matrix2[2][2] = v2.z;	matrix2[2][3] = v2.w;
	//	matrix2[3][0] = 0.0;	matrix2[3][1] = 0.0;	matrix2[3][2] = 0.0;	matrix2[3][3] = 1.0;
	//}

	//! Initializes the matrix row wise (given 4 row vectors)
	Matrix4x4(const vec4<FloatType> &v0, const vec4<FloatType> &v1, const vec4<FloatType> &v2, const vec4<FloatType> &v3) {
		matrix2[0][0] = v0.x;	matrix2[0][1] = v0.y;	matrix2[0][2] = v0.z;	matrix2[0][3] = v0.w;
		matrix2[1][0] = v1.x;	matrix2[1][1] = v1.y;	matrix2[1][2] = v1.z;	matrix2[1][3] = v1.w;
		matrix2[2][0] = v2.x;	matrix2[2][1] = v2.y;	matrix2[2][2] = v2.z;	matrix2[2][3] = v2.w;
		matrix2[3][0] = v3.x;	matrix2[3][1] = v3.y;	matrix2[3][2] = v3.z;	matrix2[3][3] = v3.w;
	}

	//! Initializes the matrix row wise
	Matrix4x4(	const FloatType &m00, const FloatType &m01, const FloatType &m02, const FloatType &m03,
				const FloatType &m10, const FloatType &m11, const FloatType &m12, const FloatType &m13,
				const FloatType &m20, const FloatType &m21, const FloatType &m22, const FloatType &m23,
				const FloatType &m30, const FloatType &m31, const FloatType &m32, const FloatType &m33) 
	{
		_m00 = m00;	_m01 = m01;	_m02 = m02;	_m03 = m03;
		_m10 = m10;	_m11 = m11;	_m12 = m12;	_m13 = m13;
		_m20 = m20;	_m21 = m21;	_m22 = m22;	_m23 = m23;
		_m30 = m30;	_m31 = m31;	_m32 = m32;	_m33 = m33;
	}

	//! Initialize with a matrix from another type
	template<class U>
	Matrix4x4(const Matrix4x4<U>& other) {
		for (unsigned int i = 0; i < 16; i++) {
			getData()[i] = (FloatType)other.getData()[i];
		}
	}

	//! Access element of Matrix at row x and column y for constant access
	inline FloatType at(unsigned char x, unsigned char y) const {
		assert((x<4)&&(y<4)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}
	//! Access element of Matrix at row x and column y
	inline FloatType& at(unsigned char x, unsigned char y) {
		assert((x<4)&&(y<4)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}

	//! Access i,j-th row of Matrix for constant access
	inline FloatType operator() (unsigned int i, unsigned int j) const {
		assert(i < 4 && j < 4);
		return matrix2[i][j];
	}
	//! Access i,j-th element of Matrix
	inline FloatType& operator() (unsigned int i, unsigned int j) {
		assert(i < 4 && j < 4);
		return matrix2[i][j]; 
	}

	//! Access i-th element of the Matrix for constant access
	inline FloatType operator[] (unsigned int i) const {
		assert(i < 16);
		return matrix[i];
	}
	//! Access i-th element of the Matrix
	inline FloatType& operator[] (unsigned int i) {
		assert(i < 16);
		return matrix[i];
	}

	//! returns the translation part of the matrix
	vec3<FloatType> getTranslation() const {
		return vec3<FloatType>(at(0,3), at(1,3), at(2,3));
	}


	//! overwrite the matrix with an identity-matrix
	void setIdentity() {
		setScale(1.0, 1.0, 1.0);
	}
	static Matrix4x4 identity() {
		Matrix4x4 res;	res.setIdentity();
		return res;
	}

	//! sets the matrix zero (or a specified value)
	void setZero(FloatType v = (FloatType)0) {
		matrix[ 0] = matrix[ 1] = matrix[ 2] = matrix[ 3] = v;
		matrix[ 4] = matrix[ 5] = matrix[ 6] = matrix[ 7] = v;
		matrix[ 8] = matrix[ 9] = matrix[10] = matrix[11] = v;
		matrix[12] = matrix[13] = matrix[14] = matrix[15] = v;
	}
	static Matrix4x4 zero(FloatType v = (FloatType)0) {
		Matrix4x4 res;	res.setZero(v);
		return res;
	}

	//! overwrite the matrix with a translation-matrix
	void setTranslation(FloatType t) {
		matrix[0] = 1.0;	matrix[1] = 0.0;	matrix[2] = 0.0; matrix[3] = t;
		matrix[4] = 0.0;	matrix[5] = 1.0;	matrix[6] = 0.0; matrix[7] = t;
		matrix[8] = 0.0;	matrix[9] = 0.0;	matrix[10] = 1.0; matrix[11] = t;
		matrix[12] = 0.0;	matrix[13] = 0.0;	matrix[14] = 0.0; matrix[15] = 1.0;
	}
	static Matrix4x4 translation(FloatType t) {
		Matrix4x4 res;	res.setTranslation(t);
		return res;
	}

	//! overwrite the matrix with a translation-matrix
	void setTranslation(FloatType x, FloatType y, FloatType z) {
		matrix[ 0] = 1.0;	matrix[ 1] = 0.0;	matrix[2]  = 0.0; matrix[3 ] = x;
		matrix[ 4] = 0.0;	matrix[ 5] = 1.0;	matrix[6]  = 0.0; matrix[7 ] = y;
		matrix[ 8] = 0.0;	matrix[ 9] = 0.0;	matrix[10] = 1.0; matrix[11] = z;
		matrix[12] = 0.0;	matrix[13] = 0.0;	matrix[14] = 0.0; matrix[15] = 1.0;
	}
	static Matrix4x4 translation(FloatType x, FloatType y, FloatType z) {
		Matrix4x4 res;	res.setTranslation(x,y,z);
		return res;
	}

	//! overwrite the matrix with a translation-matrix
	void setTranslation(const vec3<FloatType>& v) {
		matrix[0 ] = 1.0;  matrix[1 ] = 0.0;  matrix[2 ] = 0.0;  matrix[3 ] = v.x;
		matrix[4 ] = 0.0;  matrix[5 ] = 1.0;  matrix[6 ] = 0.0;  matrix[7 ] = v.y;
		matrix[8 ] = 0.0;  matrix[9 ] = 0.0;  matrix[10] = 1.0;  matrix[11] = v.z;
		matrix[12] = 0.0;  matrix[13] = 0.0;  matrix[14] = 0.0;  matrix[15] = 1.0;
	}
	static Matrix4x4 translation(const vec3<FloatType>& v) {
		Matrix4x4 res;	res.setTranslation(v);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees)
	void setRotationX(FloatType angle) {
		FloatType angleRad = math::degreesToRadians(angle);
		FloatType sinAngle = sin(angleRad);
		FloatType cosAngle = cos(angleRad);

		matrix[0 ]=1;  matrix[1 ]=0;         matrix[ 2]=0;          matrix[3 ]=0;
		matrix[4 ]=0;  matrix[5 ]=cosAngle;  matrix[ 6]=-sinAngle;  matrix[7 ]=0;
		matrix[8 ]=0;  matrix[9 ]=sinAngle;  matrix[10]= cosAngle;  matrix[11]=0;
		matrix[12]=0;  matrix[13]=0;         matrix[14]=0;          matrix[15]=1;  
	}
	static Matrix4x4 rotationX(FloatType angle) {
		Matrix4x4 res;	res.setRotationX(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees)
	void setRotationY(FloatType angle) {
		FloatType angleRad = math::degreesToRadians(angle);
		FloatType sinAngle = sin(angleRad);
		FloatType cosAngle = cos(angleRad);

		matrix[0 ]= cosAngle;  matrix[1 ]=0;  matrix[ 2]=sinAngle;  matrix[ 3]=0;
		matrix[4 ]=0;          matrix[5 ]=1;  matrix[ 6]=0;         matrix[ 7]=0;
		matrix[8 ]=-sinAngle;  matrix[9 ]=0;  matrix[10]=cosAngle;  matrix[11]=0;
		matrix[12]=0;          matrix[13]=0;  matrix[14]=0;         matrix[15]=1;   
	}
	static Matrix4x4 rotationY(FloatType angle) {
		Matrix4x4 res;	res.setRotationY(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees)
	void setRotationZ(FloatType angle) {
		FloatType angleRad = math::degreesToRadians(angle);
		FloatType sinAngle = sin(angleRad);
		FloatType cosAngle = cos(angleRad);

		matrix[0 ]=cosAngle;  matrix[1]=-sinAngle;  matrix[ 2]=0;  matrix[ 3]=0;
		matrix[4 ]=sinAngle;  matrix[5]= cosAngle;  matrix[ 6]=0;  matrix[ 7]=0;
		matrix[8 ]=0;         matrix[9]=0;          matrix[10]=1;  matrix[11]=0;
		matrix[12]=0;         matrix[13]=0;         matrix[14]=0;  matrix[15]=1;
	}
	static Matrix4x4 rotationZ(FloatType angle) {
		Matrix4x4 res;	res.setRotationZ(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees)
	void setRotation(FloatType yaw, FloatType pitch, FloatType roll) {
		*this = rotationY(yaw) * rotationX(pitch) * rotationZ(roll);
	}
	static Matrix4x4 rotation(FloatType yaw, FloatType pitch, FloatType roll) {
		Matrix4x4 res;	res.setRotation(yaw, pitch, roll);
		return res;
	}

	static Matrix4x4 rotation(const vec3<FloatType> &axis, FloatType angle) {
		Matrix4x4 res;	res.setRotation(axis, angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees)
	void setRotation(const vec3<FloatType> &axis, FloatType angle, const vec3<FloatType>& center) {
		*this = translation(-center) * rotation(axis, angle) * translation(center);
	}
	static Matrix4x4 rotation(const vec3<FloatType> &axis, FloatType angle, const vec3<FloatType>& center) {
		Matrix4x4 ret;
		ret.setRotation(axis, angle, center);
		return ret;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType x, FloatType y, FloatType z) {
		matrix[0 ] =   x; matrix[1 ] = 0.0; matrix[2 ]  = 0.0; matrix[3] = 0.0;
		matrix[4 ] = 0.0; matrix[5 ] =   y; matrix[6 ]  = 0.0; matrix[7] = 0.0;
		matrix[8 ] = 0.0; matrix[9 ] = 0.0; matrix[10] =   z; matrix[11] = 0.0;
		matrix[12] = 0.0; matrix[13] = 0.0; matrix[14] = 0.0; matrix[15] = 1.0;
	}
	static Matrix4x4 scale(FloatType x, FloatType y, FloatType z) {
		Matrix4x4 res;	res.setScale(x,y,z);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType s) {
		setScale(s,s,s);
	}
	static Matrix4x4 scale(FloatType s) {
		Matrix4x4 res;	res.setScale(s);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(const vec3<FloatType>& v) {
		matrix[0 ] = v.x; matrix[1 ] = 0.0; matrix[2 ] = 0.0; matrix[3 ] = 0.0;
		matrix[4 ] = 0.0; matrix[5 ] = v.y; matrix[6 ] = 0.0; matrix[7 ] = 0.0;
		matrix[8 ] = 0.0; matrix[9 ] = 0.0; matrix[10] = v.z; matrix[11] = 0.0;
		matrix[12] = 0.0; matrix[13] = 0.0; matrix[14] = 0.0; matrix[15] = 1.0;
	}
	static Matrix4x4 scale(const vec3<FloatType>& v) {
		Matrix4x4 res;	res.setScale(v);
		return res;
	}

    static Matrix4x4 face(const vec3<FloatType>& vA, const vec3<FloatType>& vB)
    {
        typedef vec3<FloatType> vec3;
        auto a = vA.getNormalized();
        auto b = vB.getNormalized();
        auto axis = b ^ a;
        float angle = vec3::angleBetween(a, b);

        if (angle == 0.0f) {  // No rotation
          return identity();
        } else if (axis.lengthSq() == 0.0f) {  // Need any perpendicular axis
          
          float dotX = vec3::dot(vec3::eX, a);
          if (std::abs(dotX) != (FloatType)1.0) {
            axis = vec3::eX - dotX * a;
          } else {
            axis = vec3::eY - vec3::dot(vec3::eY,a) * a;
          }
          axis.normalize();
        }
        
        return rotation(axis, -angle);
    }

	//! overwrite the matrix with a diagonal matrix
	void setDiag(FloatType x, FloatType y, FloatType z, FloatType w) {
		setScale(x,y,z);	matrix[15] = w;
	}
	static Matrix4x4 diag(FloatType x, FloatType y, FloatType z, FloatType w) {
		Matrix4x4 res;	res.setDiag(x,y,z,w);
		return res;
	}
	FloatType trace() const {
		return (matrix2[0][0] + matrix2[1][1] + matrix2[2][2] + matrix2[3][3]);
	}


	//! return the product of the operand with matrix
	Matrix4x4 operator* (const Matrix4x4& other) const {
		Matrix4x4<FloatType> result;
		//unrolling is slower (surprise?)
		for (unsigned char i = 0; i < 4; i++) {
			for (unsigned char j = 0; j < 4; j++) {
				result.at(i,j) = 
					this->at(i,0) * other.at(0,j) + 
					this->at(i,1) * other.at(1,j) + 
					this->at(i,2) * other.at(2,j) + 
					this->at(i,3) * other.at(3,j);
			}
		}
		return result;
	}

	//! multiply operand with matrix b
	Matrix4x4& operator*= (const Matrix4x4& other) {
		Matrix4x4<FloatType> prod = (*this) * other;
		*this = prod;
		return *this;
	}

	//! multiply each element in the matrix with a scalar factor
	Matrix4x4 operator* (FloatType r) const {
		Matrix4x4<FloatType> result;
		for (unsigned int i = 0; i < 16; i++) {
			result.matrix[i] = matrix[i] * r;
		}
		return result;
	}
	//! multiply each element in the matrix with a scalar factor
	Matrix4x4& operator*= (FloatType r) {
		for (unsigned int i = 0; i < 16; i++) {
			matrix[i] *= r;
		}
		return *this;
	}
	//! divide the matrix by a scalar factor
	Matrix4x4 operator/ (FloatType r) const {
		Matrix4x4<FloatType> result;
		for (unsigned int i = 0; i < 16; i++) {
			result.matrix[i] = matrix[i] / r;
		}
		return result;
	}
	//! divide each element in the matrix with a scalar factor
	Matrix4x4& operator/= (FloatType r) {
		for (unsigned int i = 0; i < 16; i++) {
			matrix[i] /= r;
		}
		return *this;
	}
	//! transform a 4D-vector with the matrix
	vec4<FloatType> operator* (const vec4<FloatType>& v) const {
		return vec4<FloatType>(
			matrix[0 ]*v.x + matrix[1 ]*v.y + matrix[2 ]*v.z + matrix[3 ]*v.w,
			matrix[4 ]*v.x + matrix[5 ]*v.y + matrix[6 ]*v.z + matrix[7 ]*v.w,
			matrix[8 ]*v.x + matrix[9 ]*v.y + matrix[10]*v.z + matrix[11]*v.w,
			matrix[12]*v.x + matrix[13]*v.y + matrix[14]*v.z + matrix[15]*v.w
		);
	}
	////! transform a 3D-vector with the matrix (implicit w=1)
	//vec4<FloatType> operator* (const vec3<FloatType>& v) const {
	//	return vec4<FloatType>(
	//		matrix[0 ]*v[0] + matrix[1 ]*v[1] + matrix[2 ]*v[2] + matrix[3],
	//		matrix[4 ]*v[0] + matrix[5 ]*v[1] + matrix[6 ]*v[2] + matrix[7],
	//		matrix[8 ]*v[0] + matrix[9 ]*v[1] + matrix[10]*v[2] + matrix[11],
	//		matrix[12]*v[0] + matrix[13]*v[1] + matrix[14]*v[2] + matrix[15]
	//	);
	//}
	//! transform a 3D-vector with the matrix (implicit w=1 and implicit subsequent de-homogenization)
	vec3<FloatType> operator* (const vec3<FloatType>& v) const {
		vec4<FloatType> result(
			matrix[0 ]*v.x + matrix[1 ]*v.y + matrix[2 ]*v.z + matrix[3 ],
			matrix[4 ]*v.x + matrix[5 ]*v.y + matrix[6 ]*v.z + matrix[7 ],
			matrix[8 ]*v.x + matrix[9 ]*v.y + matrix[10]*v.z + matrix[11],
			matrix[12]*v.x + matrix[13]*v.y + matrix[14]*v.z + matrix[15]
		);
		result.dehomogenize();
		return vec3<FloatType>(result.x, result.y, result.z);
	}

    vec3<FloatType> transformAffine(const vec3<FloatType>& v) const {
        return vec3<FloatType>(
            matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z + matrix[3],
            matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z + matrix[7],
            matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z + matrix[11]
            );
    }

    vec3<FloatType> transformNormalAffine(const vec3<FloatType>& v) const {
        return vec3<FloatType>(
            matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z,
            matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z,
            matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z
            );
    }

	//! return the sum of the operand with matrix b
	Matrix4x4 operator+ (const Matrix4x4& other) const {
		Matrix4x4<FloatType> result;
		for (unsigned int i = 0; i < 16; i++) {
			result.matrix[i] = matrix[i] + other.matrix[i];
		}
		return result;
	}

	//! add matrix other to the operand
	Matrix4x4& operator+= (const Matrix4x4& other) {
		for (unsigned int i = 0; i < 16; i++) {
			matrix[i] += other.matrix[i];
		}
		return *this;
	}

	//! return the difference of the operand with matrix b
	Matrix4x4 operator- (const Matrix4x4& other) const {
		Matrix4x4<FloatType> result;
		for (unsigned int i = 0; i < 16; i++) {
			result.matrix[i] = matrix[i] - other.matrix[i];
		}
		return result;
	}
	//! subtract matrix other from the operand
	Matrix4x4 operator-= (const Matrix4x4& other) {
		for (unsigned int i = 0; i < 16; i++) {
			matrix[i] -= other.matrix[i];
		}
		return *this;
	}

	//! return the determinant of the matrix
	FloatType det() const {
		return    matrix[0 ]*det3x3(1,2,3,1,2,3)
				- matrix[4 ]*det3x3(0,2,3,1,2,3)
				+ matrix[8 ]*det3x3(0,1,3,1,2,3)
				- matrix[12]*det3x3(0,1,2,1,2,3);
	}
	//! return the determinant of the 3x3 sub-matrix
	FloatType det3x3() const {
		return det3x3(0,1,2,0,1,2);
	}


	//! get the x column out of the matrix
	vec4<FloatType> xcol() const {
		return vec4<FloatType>(matrix[0],matrix[4],matrix[8],matrix[12]);
	}
	//! get the y column out of the matrix
	vec4<FloatType> ycol() const {
		return vec4<FloatType>(matrix[1],matrix[5],matrix[9],matrix[13]);
	}
	//! get the y column out of the matrix
	vec4<FloatType> zcol() const {
		return vec4<FloatType>(matrix[2],matrix[6],matrix[10],matrix[14]);
	}
	//! get the t column out of the matrix
	vec4<FloatType> tcol() const {
		return vec4<FloatType>(matrix[3],matrix[7],matrix[11],matrix[15]);
	}
	//! get the x row out of the matrix
	vec4<FloatType> xrow() const {
		return vec4<FloatType>(matrix[0],matrix[1],matrix[2],matrix[3]);
	}
	//! get the y row out of the matrix
	vec4<FloatType> yrow() const {
		return vec4<FloatType>(matrix[4],matrix[5],matrix[6],matrix[7]);
	}
	//! get the y row out of the matrix
	vec4<FloatType> zrow() const {
		return vec4<FloatType>(matrix[8],matrix[9],matrix[10],matrix[11]);
	}
	//! get the t row out of the matrix
	inline vec4<FloatType> trow() const {
		return vec4<FloatType>(matrix[12],matrix[13],matrix[14],matrix[15]);
	}


	//! return the inverse matrix; but does not change the current matrix
    Matrix4x4<FloatType> getInverse() const {
		FloatType inv[16];
		
		inv[0] = matrix[5]  * matrix[10] * matrix[15] - 
			matrix[5]  * matrix[11] * matrix[14] - 
			matrix[9]  * matrix[6]  * matrix[15] + 
			matrix[9]  * matrix[7]  * matrix[14] +
			matrix[13] * matrix[6]  * matrix[11] - 
			matrix[13] * matrix[7]  * matrix[10];

		inv[4] = -matrix[4]  * matrix[10] * matrix[15] + 
			matrix[4]  * matrix[11] * matrix[14] + 
			matrix[8]  * matrix[6]  * matrix[15] - 
			matrix[8]  * matrix[7]  * matrix[14] - 
			matrix[12] * matrix[6]  * matrix[11] + 
			matrix[12] * matrix[7]  * matrix[10];

		inv[8] = matrix[4]  * matrix[9] * matrix[15] - 
			matrix[4]  * matrix[11] * matrix[13] - 
			matrix[8]  * matrix[5] * matrix[15] + 
			matrix[8]  * matrix[7] * matrix[13] + 
			matrix[12] * matrix[5] * matrix[11] - 
			matrix[12] * matrix[7] * matrix[9];

		inv[12] = -matrix[4]  * matrix[9] * matrix[14] + 
			matrix[4]  * matrix[10] * matrix[13] +
			matrix[8]  * matrix[5] * matrix[14] - 
			matrix[8]  * matrix[6] * matrix[13] - 
			matrix[12] * matrix[5] * matrix[10] + 
			matrix[12] * matrix[6] * matrix[9];

		inv[1] = -matrix[1]  * matrix[10] * matrix[15] + 
			matrix[1]  * matrix[11] * matrix[14] + 
			matrix[9]  * matrix[2] * matrix[15] - 
			matrix[9]  * matrix[3] * matrix[14] - 
			matrix[13] * matrix[2] * matrix[11] + 
			matrix[13] * matrix[3] * matrix[10];

		inv[5] = matrix[0]  * matrix[10] * matrix[15] - 
			matrix[0]  * matrix[11] * matrix[14] - 
			matrix[8]  * matrix[2] * matrix[15] + 
			matrix[8]  * matrix[3] * matrix[14] + 
			matrix[12] * matrix[2] * matrix[11] - 
			matrix[12] * matrix[3] * matrix[10];

		inv[9] = -matrix[0]  * matrix[9] * matrix[15] + 
			matrix[0]  * matrix[11] * matrix[13] + 
			matrix[8]  * matrix[1] * matrix[15] - 
			matrix[8]  * matrix[3] * matrix[13] - 
			matrix[12] * matrix[1] * matrix[11] + 
			matrix[12] * matrix[3] * matrix[9];

		inv[13] = matrix[0]  * matrix[9] * matrix[14] - 
			matrix[0]  * matrix[10] * matrix[13] - 
			matrix[8]  * matrix[1] * matrix[14] + 
			matrix[8]  * matrix[2] * matrix[13] + 
			matrix[12] * matrix[1] * matrix[10] - 
			matrix[12] * matrix[2] * matrix[9];

		inv[2] = matrix[1]  * matrix[6] * matrix[15] - 
			matrix[1]  * matrix[7] * matrix[14] - 
			matrix[5]  * matrix[2] * matrix[15] + 
			matrix[5]  * matrix[3] * matrix[14] + 
			matrix[13] * matrix[2] * matrix[7] - 
			matrix[13] * matrix[3] * matrix[6];

		inv[6] = -matrix[0]  * matrix[6] * matrix[15] + 
			matrix[0]  * matrix[7] * matrix[14] + 
			matrix[4]  * matrix[2] * matrix[15] - 
			matrix[4]  * matrix[3] * matrix[14] - 
			matrix[12] * matrix[2] * matrix[7] + 
			matrix[12] * matrix[3] * matrix[6];

		inv[10] = matrix[0]  * matrix[5] * matrix[15] - 
			matrix[0]  * matrix[7] * matrix[13] - 
			matrix[4]  * matrix[1] * matrix[15] + 
			matrix[4]  * matrix[3] * matrix[13] + 
			matrix[12] * matrix[1] * matrix[7] - 
			matrix[12] * matrix[3] * matrix[5];

		inv[14] = -matrix[0]  * matrix[5] * matrix[14] + 
			matrix[0]  * matrix[6] * matrix[13] + 
			matrix[4]  * matrix[1] * matrix[14] - 
			matrix[4]  * matrix[2] * matrix[13] - 
			matrix[12] * matrix[1] * matrix[6] + 
			matrix[12] * matrix[2] * matrix[5];

		inv[3] = -matrix[1] * matrix[6] * matrix[11] + 
			matrix[1] * matrix[7] * matrix[10] + 
			matrix[5] * matrix[2] * matrix[11] - 
			matrix[5] * matrix[3] * matrix[10] - 
			matrix[9] * matrix[2] * matrix[7] + 
			matrix[9] * matrix[3] * matrix[6];

		inv[7] = matrix[0] * matrix[6] * matrix[11] - 
			matrix[0] * matrix[7] * matrix[10] - 
			matrix[4] * matrix[2] * matrix[11] + 
			matrix[4] * matrix[3] * matrix[10] + 
			matrix[8] * matrix[2] * matrix[7] - 
			matrix[8] * matrix[3] * matrix[6];

		inv[11] = -matrix[0] * matrix[5] * matrix[11] + 
			matrix[0] * matrix[7] * matrix[9] + 
			matrix[4] * matrix[1] * matrix[11] - 
			matrix[4] * matrix[3] * matrix[9] - 
			matrix[8] * matrix[1] * matrix[7] + 
			matrix[8] * matrix[3] * matrix[5];

		inv[15] = matrix[0] * matrix[5] * matrix[10] - 
			matrix[0] * matrix[6] * matrix[9] - 
			matrix[4] * matrix[1] * matrix[10] + 
			matrix[4] * matrix[2] * matrix[9] + 
			matrix[8] * matrix[1] * matrix[6] - 
			matrix[8] * matrix[2] * matrix[5];

		FloatType matrixDet = matrix[0] * inv[0] + matrix[1] * inv[4] + matrix[2] * inv[8] + matrix[3] * inv[12];
		
		FloatType matrixDetr = (FloatType)1.0 / matrixDet;

		Matrix4x4<FloatType> res;
		for (unsigned int i = 0; i < 16; i++) {
			res.matrix[i] = inv[i] * matrixDetr;
		}
		return res;

	}

	const FloatType* getData() const
	{
		return matrix;
	}

	FloatType* getData()
	{
		return matrix;
	}

	//! overwrite the current matrix with its inverse
	void invert() {
		*this = getInverse();
	}
	//! return the transposed matrix
	Matrix4x4 getTranspose() const {
		Matrix4x4<FloatType> result;
		for(unsigned char x = 0; x < 4; x++) {
			result.at(x,0) = at(0,x);
			result.at(x,1) = at(1,x);
			result.at(x,2) = at(2,x);
			result.at(x,3) = at(3,x);
		}
		return result;
	}
	//! transpose the matrix in place
	void transpose() {
		*this = getTranspose();
	}

	//! returns true if the matrix is affine (i.e., projective part is zero)
	bool isAffine(FloatType eps = (FloatType)0.000001) const {
		if (math::floatEqual<FloatType>(matrix[12], 0, eps) && 
			math::floatEqual<FloatType>(matrix[13], 0, eps) && 
			math::floatEqual<FloatType>(matrix[14], 0, eps) && 
			math::floatEqual<FloatType>(matrix[15], 1, eps))	
			return true;
		else 
			return false;
	}
	
	unsigned int rank(FloatType eps = (FloatType)0.00001) const {
		return util::rank<Matrix4x4<FloatType>, FloatType>(*this, 4, eps);
	}

	  union {
		  //! access matrix using a single array
		  FloatType matrix[16];
		  //! access matrix using a two-dimensional array
		  FloatType matrix2[4][4];
		  //! access matrix using single elements
		  struct {
			  FloatType
				  _m00, _m01, _m02, _m03,
				  _m10, _m11, _m12, _m13,
				  _m20, _m21, _m22, _m23,
				  _m30, _m31, _m32, _m33;
		  };
	  };

  private:
    //! calculate determinant of a 3x3 sub-matrix given by the indices of the rows and columns
    FloatType det3x3(unsigned int i0 = 0, unsigned int i1 = 1, unsigned int i2 = 2, unsigned int j0 = 0, unsigned int j1 = 1, unsigned int j2 = 2) const {
      return
        (matrix2[i0][j0]*matrix2[i1][j1]*matrix2[i2][j2])
        + (matrix2[i0][j1]*matrix2[i1][j2]*matrix2[i2][j0])
        + (matrix2[i0][j2]*matrix2[i1][j0]*matrix2[i2][j1])
        - (matrix2[i2][j0]*matrix2[i1][j1]*matrix2[i0][j2])
        - (matrix2[i2][j1]*matrix2[i1][j2]*matrix2[i0][j0])
        - (matrix2[i2][j2]*matrix2[i1][j0]*matrix2[i0][j1]);
    }
	////! quaternion is a friend, because it should be able to instance an uninitialized matrix (Matrix4x4 Quaternion::matrix())
	//friend class Quaternion<FloatType>;
};

//! writes to a stream
template <class FloatType> 
inline std::ostream& operator<<(std::ostream& s, const Matrix4x4<FloatType>& m)
{ 
	return (
		s << 
		m(0,0) << " " << m(0,1) << " " << m(0,2) << " " << m(0,3) << std::endl <<
		m(1,0) << " " << m(1,1) << " " << m(1,2) << " " << m(1,3) << std::endl <<
		m(2,0) << " " << m(2,1) << " " << m(2,2) << " " << m(2,3) << std::endl <<
		m(3,0) << " " << m(3,1) << " " << m(3,2) << " " << m(3,3) << std::endl
		);
}

//! reads from a stream
template <class FloatType> 
inline std::istream& operator>>(std::istream& s, const Matrix4x4<FloatType>& m)
{ 
	return (
		s >> 
		m(0,0) >> m(0,1) >> m(0,2)  >> m(0,3) >>
		m(1,0) >> m(1,1) >> m(1,2)  >> m(1,3) >>
		m(2,0) >> m(2,1) >> m(2,2)  >> m(2,3) >>
		m(3,0) >> m(3,1) >> m(3,2)  >> m(3,3));
}




typedef Matrix4x4<int> mat4i;
typedef Matrix4x4<int> mat4u;
typedef Matrix4x4<float> mat4f;
typedef Matrix4x4<double> mat4d;

}  // namespace ml

#endif  // CORE_MATH_MATRIX4X4_H_
