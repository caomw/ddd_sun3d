#include "pc2tsdf/pc2tsdf.h"
#include "detect_keypoints.h"
#include "ddd.h"

// float gen_random_float(float min, float max) {
//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<double> dist(min, max - 0.0001);
//   return dist(mt);
// }

int main() {

  /* DEPRECATED

  const float voxelSize = 0.01f;
  const float truncationRadius = 0.05f;

  // // Init extrinsic matrix to identity
  // float *ext_mat = new float[16];
  // for (int i = 0; i < 16; i++)
  //   ext_mat[i] = 0.0f;
  // ext_mat[0] = 1.0f;
  // ext_mat[5] = 1.0f;
  // ext_mat[10] = 1.0f;
  // ext_mat[15] = 1.0f;

  std::string data_directory = "/data/andyz/kinfu/data/augICLNUIMDataset/fragments/livingroom1-fragments-ply/";

  for (int cloud_idx = 0; cloud_idx < 56; cloud_idx++) {

    std::string pc_filename1 = data_directory + "cloud_bin_" + std::to_string(cloud_idx) + ".ply";
    std::string pc_filename2 = data_directory + "cloud_bin_" + std::to_string(cloud_idx + 1) + ".ply";

    ///////////////////////////////////////////////////////////////////

    // Load first point cloud and save to TUDF grid data
    auto cloud1 = PointCloudIOf::loadFromFile(pc_filename1);
    pc2tsdf::TSDF tsdf1;
    pc2tsdf::makeTSDF(cloud1, voxelSize, truncationRadius, tsdf1);
    // tsdf1.saveBinary("testFragmentFixed.tsdf");

    // Convert TUDF grid data to float array
    int x_dim1 = (int)tsdf1.data.getDimensions().x;
    int y_dim1 = (int)tsdf1.data.getDimensions().y;
    int z_dim1 = (int)tsdf1.data.getDimensions().z;
    float *scene_tsdf1 = new float[x_dim1 * y_dim1 * z_dim1];
    for (int i = 0; i < x_dim1 * y_dim1 * z_dim1; i++)
      scene_tsdf1[i] = 1.0;
    for (int z = 0; z < (int)tsdf1.data.getDimensions().z; z++)
      for (int y = 0; y < (int)tsdf1.data.getDimensions().y; y++)
        for (int x = 0; x < (int)tsdf1.data.getDimensions().x; x++)
          scene_tsdf1[z * y_dim1 * x_dim1 + y * x_dim1 + x] = tsdf1.data(x, y, z) / truncationRadius;

    // Load second point cloud and save to TUDF
    auto cloud2 = PointCloudIOf::loadFromFile(pc_filename2);
    pc2tsdf::TSDF tsdf2;
    pc2tsdf::makeTSDF(cloud2, voxelSize, truncationRadius, tsdf2);

    // Convert TUDF grid data to float array
    int x_dim2 = (int)tsdf2.data.getDimensions().x;
    int y_dim2 = (int)tsdf2.data.getDimensions().y;
    int z_dim2 = (int)tsdf2.data.getDimensions().z;
    float *scene_tsdf2 = new float[x_dim2 * y_dim2 * z_dim2];
    for (int i = 0; i < x_dim2 * y_dim2 * z_dim2; i++)
      scene_tsdf2[i] = 1.0;
    for (int z = 0; z < (int)tsdf2.data.getDimensions().z; z++)
      for (int y = 0; y < (int)tsdf2.data.getDimensions().y; y++)
        for (int x = 0; x < (int)tsdf2.data.getDimensions().x; x++)
          scene_tsdf2[z * y_dim2 * x_dim2 + y * x_dim2 + x] = tsdf2.data(x, y, z) / truncationRadius;

    ///////////////////////////////////////////////////////////////////

    float k_match_score_thresh = 0.5f;
    float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
    float max_ransac_iter = 1000000;
    float ransac_inlier_thresh = 0.01f;
    float* Rt = new float[12]; // Contains rigid transform matrix
    align2tsdf(scene_tsdf1, x_dim1, y_dim1, z_dim1, tsdf1.worldOrigin[0], tsdf1.worldOrigin[1], tsdf1.worldOrigin[2],
               scene_tsdf2, x_dim2, y_dim2, z_dim2, tsdf2.worldOrigin[0], tsdf2.worldOrigin[1], tsdf2.worldOrigin[2], 
               voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

    ///////////////////////////////////////////////////////////////////

    // Apply Rt to second point cloud and align it to first
    for (int i = 0; i < cloud2.m_points.size(); i++) {
      vec3f tmp_point;
      tmp_point.x = Rt[0] * cloud2.m_points[i].x + Rt[1] * cloud2.m_points[i].y + Rt[2] * cloud2.m_points[i].z;
      tmp_point.y = Rt[4] * cloud2.m_points[i].x + Rt[5] * cloud2.m_points[i].y + Rt[6] * cloud2.m_points[i].z;
      tmp_point.z = Rt[8] * cloud2.m_points[i].x + Rt[9] * cloud2.m_points[i].y + Rt[10] * cloud2.m_points[i].z;
      tmp_point.x = tmp_point.x + Rt[3];
      tmp_point.y = tmp_point.y + Rt[7];
      tmp_point.z = tmp_point.z + Rt[11];
      cloud2.m_points[i] = tmp_point;
    }

    // Make point clouds colorful
    // ml::vec3f color1;
    // for (int i = 0; i < 3; i++)
    //   color1[i] = gen_random_float(0.0, 1.0);
    // for (int i = 0; i < cloud1.m_points.size(); i++)
    //   cloud1.m_colors[i] = color1;

    // ml::vec3f color2;
    // for (int i = 0; i < 3; i++)
    //   color2[i] = gen_random_float(0.0, 1.0);
    // for (int i = 0; i < cloud2.m_points.size(); i++)
    //   cloud2.m_colors[i] = color2;

    // std::cout << cloud1.m_points.size() << std::endl;
    // std::cout << cloud1.m_normals.size() << std::endl;
    // std::cout << cloud1.m_colors.size() << std::endl;

    // for (int i = 0; i < cloud1.m_colors.size(); i++) {
    //   std::cout << cloud1.m_colors[i].x << " " << cloud1.m_colors[i].y << " " << cloud1.m_colors[i].z << std::endl;
    // }

    // cloud1.m_colors.clear();
    // cloud2.m_colors.clear();

    // Print out both point clouds
    // if (cloud_idx == 0) {
    std::string pcfile1 = "test" + std::to_string(cloud_idx) + "_" + std::to_string(cloud_idx + 1) + "_" + std::to_string(cloud_idx) + ".ply";
    PointCloudIOf::saveToFile(pcfile1, cloud1);
    // }
    std::string pcfile2 = "test" + std::to_string(cloud_idx) + "_" + std::to_string(cloud_idx + 1) + "_" + std::to_string(cloud_idx + 1) + ".ply";
    PointCloudIOf::saveToFile(pcfile2, cloud2);
    // >/dev/null

    delete [] scene_tsdf1;
    delete [] scene_tsdf2;
    delete [] Rt;
  }

  */

  return 0;
}
