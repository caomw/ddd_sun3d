#include <vector>

///////////////////////////////////////////////////////////////////////
// Given a location (x, y, z) in the voxel volume, compute its pseudo-
// surface normal, which is simply the voxel value gradients across
// the X,Y,Z directions.
// Parameters:
//   volume - dense array of floats to represent voxel volume
//            volume[z*y_dim*x_dim + y*x_dim + x] --> volume(x,y,z)
//   x_dim, y_dim, z_dim - voxel volume size in X,Y,Z dimensions
//   x, y, z - voxel location to compute normal
//   weight  - multiply the computed normal by this weight value
//
// Returns a normal as a vector of three floats
//
// Copyright (c) 2015 Andy Zeng, Princeton University
std::vector<float> compute_norm(float* volume, int x_dim, int y_dim, int z_dim, int x, int y, int z, float weight) {

  // Compute voxel value gradients
  float SDFx1 = volume[(z) * y_dim * x_dim + (y) * x_dim + (x + 1)];
  float SDFx2 = volume[(z) * y_dim * x_dim + (y) * x_dim + (x - 1)];
  float SDFy1 = volume[(z) * y_dim * x_dim + (y + 1) * x_dim + (x)];
  float SDFy2 = volume[(z) * y_dim * x_dim + (y - 1) * x_dim + (x)];
  float SDFz1 = volume[(z + 1) * y_dim * x_dim + (y) * x_dim + (x)];
  float SDFz2 = volume[(z - 1) * y_dim * x_dim + (y) * x_dim + (x)];

  // Use gradients as pseudo surface normal
  std::vector<float> normal = {SDFx1 - SDFx2, SDFy1 - SDFy2, SDFz1 - SDFz2};

  if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
    return normal;

  // Normalize
  float denom = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  normal[0] = weight * normal[0] / denom;
  normal[1] = weight * normal[1] / denom;
  normal[2] = weight * normal[2] / denom;
  return normal;

}

///////////////////////////////////////////////////////////////////////
// Given a location (x, y, z) in the voxel volume, compute the covariance
// matrix over the surface normals the local patch volume around (x, y, z)
//
// Parameters:
//   volume - dense array of floats to represent voxel volume
//            volume[z*y_dim*x_dim + y*x_dim + x] --> volume(x,y,z)
//   x_dim, y_dim, z_dim - voxel volume size in X,Y,Z dimensions
//   x, y, z - voxel location of center of local patch volume
//   tsdf_threshold      - compute surface normals only on voxels
//                         with a abs(value) < tsdf_threshold
//   radius              - specify radius (in voxels) of local
//                         patch volume
//   norm_scale          - scale the computed normals by norm_scale
//   covariance          - 3x3 row-major matrix
//
// Covariance matrix is stored in parameter "covariance"
// Function returns true if covariance was successfully computed
//
// Copyright (c) 2015 Andy Zeng, Princeton University
bool compute_norm_covar(float* volume, int x_dim, int y_dim, int z_dim, int x, int y, int z, float tsdf_threshold, int radius, float norm_scale, float* covariance) {

  // Each normal is a vector three floats
  std::vector<std::vector<float>> normals;

  // Compute normals for every voxel within 'radius' voxels
  for (int k = -radius; k <= radius; k++)
    for (int j = -radius; j <= radius; j++)
      for (int i = -radius; i <= radius; i++)
        // Compute normals only on near-surface voxels
        if (std::abs(volume[(z + k) * y_dim * x_dim + (y + j) * x_dim + (x + i)]) < tsdf_threshold) {
          // Weight normals with gaussian kernel
          float ksize = (float)radius * 2.0f + 1.0f;
          float sigma = 0.3f * (ksize / 2.0f - 1.0f) + 0.8f;
          float inc_dist = std::max(std::abs((float)i), std::max(std::abs((float)j), std::abs((float)k)));
          float gauss_w = exp(-(inc_dist * inc_dist) / (2.0f * sigma * sigma)) / (sqrt(2.0f * 3.1415927) * sigma);
          // Add normal computed on voxel
          std::vector<float> tmp_norm = compute_norm(volume, x_dim, y_dim, z_dim, x + i, y + j, z + k, gauss_w * norm_scale);
          normals.push_back(tmp_norm);
        }

  // Skip computing covariance if insufficient relevant normals
  if (normals.size() < (radius * 2 + 1) * (radius * 2 + 1))
    return false;

  // Compute covariance matrix over normals
  float Ixx = 0; float Ixy = 0; float Ixz = 0;
  float Iyy = 0; float Iyz = 0; float Izz = 0;
  for (int i = 0; i < normals.size(); i++) {
    Ixx = Ixx + normals[i][0] * normals[i][0];
    Ixy = Ixy + normals[i][0] * normals[i][1];
    Ixz = Ixz + normals[i][0] * normals[i][2];
    Iyy = Iyy + normals[i][1] * normals[i][1];
    Iyz = Iyz + normals[i][1] * normals[i][2];
    Izz = Izz + normals[i][2] * normals[i][2];
  }
  float scale = 1.0f / ((float)normals.size());
  covariance[0] = Ixx * scale;
  covariance[1] = Ixy * scale;
  covariance[2] = Ixz * scale;
  covariance[3] = Ixy * scale;
  covariance[4] = Iyy * scale;
  covariance[5] = Iyz * scale;
  covariance[6] = Ixz * scale;
  covariance[7] = Iyz * scale;
  covariance[8] = Izz * scale;

  return true;
}

///////////////////////////////////////////////////////////////////////
// Given a TSDF/TUDF voxel volume, compute harris keypoints using
// the normals computed from voxel value gradients.
//
// Parameters:
//   volume - dense array of floats to represent voxel volume
//            volume[z*y_dim*x_dim + y*x_dim + x] --> volume(x,y,z)
//   x_dim, y_dim, z_dim - voxel volume size in X,Y,Z dimensions
//   tsdf_threshold      - compute harris response only on voxels
//                         with a abs(value) < tsdf_threshold
//   response_threshold  - keypoints need to have a response that
//                         is > response_threshold
//   radius              - specify radius (in voxels) of harris
//                         patch; this is also the radius for
//                         local maximum search
//   norm_scale          - scale the computed normals by norm_scale
//
// Returns a vector of keypoints:
//   Each keypoint is a vector of three integers: (x,y,z).
//
// Copyright (c) 2015 Andy Zeng, Princeton University
std::vector<std::vector<int>> detect_keypoints(float *volume, int x_dim, int y_dim, int z_dim, float tsdf_threshold, float response_threshold, int radius, float norm_scale) {

  // Init harris reponses of volume to 0
  float* responses = new float[x_dim * y_dim * z_dim];
  memset(responses, 0, sizeof(float) * x_dim * y_dim * z_dim);
  std::vector<std::vector<int>> keypoints;

  // Compute harris responses
  for (int z = radius + 1; z < z_dim - radius - 1; z++)
    for (int y = radius + 1; y < y_dim - radius - 1; y++)
      for (int x = radius + 1; x < x_dim - radius - 1; x++) {
        int voxel_idx = z * y_dim * x_dim + y * x_dim + x;
        // Compute harris response only on near-surface voxels
        if (std::abs(volume[voxel_idx]) < tsdf_threshold) {
          // Compute covariance matrix from normals
          float covar[9];
          bool isValid = compute_norm_covar(volume, x_dim, y_dim, z_dim, x, y, z, tsdf_threshold, radius, norm_scale, covar);
          float response = 0;
          if (isValid) {
            // Use covariance matrix to ompute harris corner response
            float covar_det = covar[0] * covar[4] * covar[8] + covar[1] * covar[5] * covar[6] + covar[2] * covar[3] * covar[7] -
                              covar[6] * covar[4] * covar[2] - covar[7] * covar[5] * covar[0] - covar[8] * covar[3] * covar[1];
            float covar_trace = covar[0] + covar[4] + covar[8];
            if (covar_trace != 0)
              response = 0.04f + covar_det - 0.04f * covar_trace * covar_trace;
            // DEBUG: std::cout << response << std::endl;
          }
          responses[voxel_idx] = response;
        }
      }

  // Search for keypoints = local maximum over harris responses
  for (int z = radius + 1; z < z_dim - radius - 1; z++)
    for (int y = radius + 1; y < y_dim - radius - 1; y++)
      for (int x = radius + 1; x < x_dim - radius - 1; x++) {
        int voxel_idx = z * y_dim * x_dim + y * x_dim + x;
        // Skip voxels with low harris response
        if (responses[voxel_idx] < response_threshold)
          continue;
        // Check if voxel is a local maximum within 'radius' voxels
        bool isCorner = true;
        for (int k = -radius; k <= radius; k++)
          for (int j = -radius; j <= radius; j++)
            for (int i = -radius; i <= radius; i++) {
              if (isCorner && responses[(z + k) * y_dim * x_dim + (y + j) * x_dim + (x + i)] > responses[voxel_idx])
                isCorner = false;
            }
        // If point is a local maximum, add as keypoint
        if (isCorner) {
          std::vector<int> tmp_keypoint = {x, y, z};
          keypoints.push_back(tmp_keypoint);
        }
      }

  delete [] responses;
  return keypoints;
}