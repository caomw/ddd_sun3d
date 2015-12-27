#include <kinfu.hpp>

int main(int argc, char **argv) {


  bool ddd_verbose = true;
  const float k_match_score_thresh = 0.5f;
  const float ransac_k = 3; // RANSAC over top-k > k_match_score_thresh
  const float max_ransac_iter = 1000000;
  const float ransac_inlier_thresh = 4.0f; // distance in grid coordinates, default: 4 voxels

  // Fuse first fragment
  int fuse_frame_start_idx = 600;
  int fuse_frame_end_idx = 650;
  int num_frames_per_frag = 50;
  std::string frag_saveto_dir = "/data/andyz/kinfu/sun3d/mit_32_d507_d507_2/";
  std::string sun3d_data_load_dir = "/data/andyz/kinfu/data/sun3d/";
  std::string scene1_dir = "/data/andyz/kinfu/sun3d/mit_32_d507_d507_2/scene600_650";
  generate_data_sun3d(frag_saveto_dir, sun3d_data_load_dir, fuse_frame_start_idx, fuse_frame_end_idx, num_frames_per_frag);

  // Fuse second fragment
  fuse_frame_start_idx = 2100;
  fuse_frame_end_idx = 2150;
  num_frames_per_frag = 50;
  frag_saveto_dir = "/data/andyz/kinfu/sun3d/mit_32_d507_d507_2/";
  sun3d_data_load_dir = "/data/andyz/kinfu/data/sun3d/";
  std::string scene2_dir = "/data/andyz/kinfu/sun3d/mit_32_d507_d507_2/scene2100_2150";
  generate_data_sun3d(frag_saveto_dir, sun3d_data_load_dir, fuse_frame_start_idx, fuse_frame_end_idx, num_frames_per_frag);

  // Load first fragment TSDF
  float *scene_tsdf1 = new float[512 * 512 * 1024];
  checkout_tsdf(scene1_dir, scene_tsdf1, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  std::vector<keypoint> keypoints1;
  checkout_keypts(scene1_dir, keypoints1);

  // Convert from vector of keypoints to vector of vectors
  std::vector<std::vector<int>> keypoints1_c;
  std::vector<std::vector<float>> keypoints1_w;
  for (int i = 0; i < keypoints1.size(); i++) {
    std::vector<int> tmp_keypoint;
    tmp_keypoint.push_back((int) std::round(keypoints1[i].x));
    tmp_keypoint.push_back((int) std::round(keypoints1[i].y));
    tmp_keypoint.push_back((int) std::round(keypoints1[i].z));
    keypoints1_c.push_back(tmp_keypoint);
    std::vector<float> tmp_keypoint_w;
    tmp_keypoint_w.push_back(keypoints1[i].x);
    tmp_keypoint_w.push_back(keypoints1[i].y);
    tmp_keypoint_w.push_back(keypoints1[i].z);
    keypoints1_w.push_back(tmp_keypoint_w);
  }

  // Load second fragment TSDF
  float *scene_tsdf2 = new float[512 * 512 * 1024];
  checkout_tsdf(scene2_dir, scene_tsdf2, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  std::vector<keypoint> keypoints2;
  checkout_keypts(scene2_dir, keypoints2);

  // Convert from vector of keypoints to vector of vectors
  std::vector<std::vector<int>> keypoints2_c;
  std::vector<std::vector<float>> keypoints2_w;
  for (int i = 0; i < keypoints2.size(); i++) {
    std::vector<int> tmp_keypoint;
    tmp_keypoint.push_back((int) std::round(keypoints2[i].x));
    tmp_keypoint.push_back((int) std::round(keypoints2[i].y));
    tmp_keypoint.push_back((int) std::round(keypoints2[i].z));
    keypoints2_c.push_back(tmp_keypoint);
    std::vector<float> tmp_keypoint_w;
    tmp_keypoint_w.push_back(keypoints2[i].x);
    tmp_keypoint_w.push_back(keypoints2[i].y);
    tmp_keypoint_w.push_back(keypoints2[i].z);
    keypoints2_w.push_back(tmp_keypoint_w);
  }

  // Compute feature vectors from local keypoint patches
  std::vector<std::vector<float>> feat1;  
  feat1 = ddd_get_keypoint_feat(scene_tsdf1, 512, 512, 1024, keypoints1_c, 15, ddd_verbose);
  std::vector<std::vector<float>> feat2;  
  feat2 = ddd_get_keypoint_feat(scene_tsdf2, 512, 512, 1024, keypoints2_c, 15, ddd_verbose);

  // Compare feature vectors and compute score matrix
  std::vector<std::vector<float>> score_matrix1;  
  score_matrix1 = ddd_compare_feat(feat1, feat2, ddd_verbose);

  // For each keypoint from first set, find indices of all keypoints
  // in second set with score > k_match_score_thresh
  std::vector<std::vector<int>> match_rank1;
  for (int i = 0; i < feat1.size(); i++) {
    // Sort score vector in descending fashion
    std::vector<float> tmp_score_vect = score_matrix1[i];
    float* tmp_score_vect_arr = &tmp_score_vect[0];
    int* tmp_score_idx = new int[tmp_score_vect.size()];
    std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
    std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
    std::vector<int> tmp_score_rank;
    for (int j = 0; j < feat2.size(); j++)
      if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
        tmp_score_rank.push_back(tmp_score_idx[j]);
    // std::cout << tmp_score_rank.size() << std::endl;
    match_rank1.push_back(tmp_score_rank);
  }

  // Inversely compare feature vectors and compute score matrix
  std::vector<std::vector<float>> score_matrix2;  
  score_matrix2 = ddd_compare_feat(feat2, feat1, ddd_verbose);

  // For each keypoint from second set, find indices of all keypoints
  // in first set with score > k_match_score_thresh
  std::vector<std::vector<int>> match_rank2;
  for (int i = 0; i < feat2.size(); i++) {
    // Sort score vector in descending fashion
    std::vector<float> tmp_score_vect = score_matrix2[i];
    float* tmp_score_vect_arr = &tmp_score_vect[0];
    int* tmp_score_idx = new int[tmp_score_vect.size()];
    std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
    std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
    std::vector<int> tmp_score_rank;
    for (int j = 0; j < feat1.size(); j++)
      if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
        tmp_score_rank.push_back(tmp_score_idx[j]);
    // std::cout << tmp_score_rank.size() << std::endl;
    match_rank2.push_back(tmp_score_rank);
  }

  // Finalize match matrix (indices) unofficial reflexive property
  // A pair of points (with feature vectors f1 and f2) match iff
  // ddd(f1,f2) > threshold && ddd(f2,f1) > threshold
  std::vector<std::vector<int>> match_idx;
  for (int i = 0; i < feat1.size(); i++) {
    std::vector<int> tmp_matches;
    for (int j = 0; j < match_rank1[i].size(); j++) {
      int tmp_match_idx = match_rank1[i][j];
      if (std::find(match_rank2[tmp_match_idx].begin(), match_rank2[tmp_match_idx].end(), i) != match_rank2[tmp_match_idx].end())
        tmp_matches.push_back(tmp_match_idx);
    }
    match_idx.push_back(tmp_matches);
  }

  // DEBUG
  if (ddd_verbose) {
    for (int i = 0; i < feat1.size(); i++) {
    std::cout << i << " | ";
    for (int j = 0; j < match_idx[i].size(); j++)
        std::cout << match_idx[i][j] << " ";
    std::cout << std::endl;
    }
  }

  // Compute Rt transform from second to first point cloud (k-ransac)
  float* Rt = new float[16]; 
  Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;
  int num_inliers = ransacfitRt(keypoints1_w, keypoints2_w, match_idx, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt, ddd_verbose);

  for (int i = 0; i < 16; i++)
    std::cout << Rt[i] << std::endl;

  // Create point cloud for first tsdf
  float tsdf_threshold = 0.2f;
  int num_points = 0;
  for (int i = 0; i < 512*512*1024; i++)
    if (scene_tsdf1[i] < tsdf_threshold)
      num_points++;
  FILE *fp = fopen("pointcloud1.ply", "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");
  for (int z = 0; z < 1024; z++)
    for (int y = 0; y < 512; y++)
      for (int x = 0; x < 512; x++)
        if (scene_tsdf1[z*512*512 + y*512 +x] < tsdf_threshold) {
          float float_x = (float) x;
          float float_y = (float) y;
          float float_z = (float) z;
          fwrite(&float_x, sizeof(float), 1, fp);
          fwrite(&float_y, sizeof(float), 1, fp);
          fwrite(&float_z, sizeof(float), 1, fp);
          unsigned char r = (unsigned char )255;
          unsigned char g = (unsigned char )0;
          unsigned char b = (unsigned char )0;
          fwrite(&r, sizeof(uchar), 1, fp);
          fwrite(&g, sizeof(uchar), 1, fp);
          fwrite(&b, sizeof(uchar), 1, fp);
        }
  fclose(fp);

  // Create point cloud for second tsdf
  num_points = 0;
  for (int i = 0; i < 512*512*1024; i++)
    if (scene_tsdf2[i] < tsdf_threshold)
      num_points++;
  fp = fopen("pointcloud2.ply", "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");
  for (int z = 0; z < 1024; z++)
    for (int y = 0; y < 512; y++)
      for (int x = 0; x < 512; x++)
        if (scene_tsdf2[z*512*512 + y*512 +x] < tsdf_threshold) {
          float float_x = (float) x;
          float float_y = (float) y;
          float float_z = (float) z;
          float trans_x = Rt[0] * float_x + Rt[1] * float_y + Rt[2] * float_z + Rt[3];
          float trans_y = Rt[4] * float_x + Rt[5] * float_y + Rt[6] * float_z + Rt[7];
          float trans_z = Rt[8] * float_x + Rt[9] * float_y + Rt[10] * float_z + Rt[11];
          fwrite(&trans_x, sizeof(float), 1, fp);
          fwrite(&trans_y, sizeof(float), 1, fp);
          fwrite(&trans_z, sizeof(float), 1, fp);
          unsigned char r = (unsigned char )0;
          unsigned char g = (unsigned char )0;
          unsigned char b = (unsigned char )255;
          fwrite(&r, sizeof(uchar), 1, fp);
          fwrite(&g, sizeof(uchar), 1, fp);
          fwrite(&b, sizeof(uchar), 1, fp);
        }
  fclose(fp);



  // float tsdf_threshold = 0.2;
  // int x_dim = 512; int y_dim = 512; int z_dim = 1024;
  // std::string ply_filename1 = "test_scene0_30.ply";
  // float *ext_mat = new float[16];
  // checkout_ext("scene0_30", ext_mat);
  // tsdf2ply(ply_filename1, scene1_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);

  // float *scene2_tsdf = new float[512 * 512 * 1024];
  // checkout_tsdf(scene2_dir, scene2_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  // std::vector<keypoint> keypoints2;
  // checkout_keypts(scene2_dir, keypoints2);

// ///////////////////////////////////////////////////////////////////

//         const float k_match_score_thresh = 0.5f;
//         const float ransac_k = 3; // RANSAC over top-k > k_match_score_thresh
//         const float max_ransac_iter = 500000;
//         const float ransac_inlier_thresh = 0.04f;
        
//         float* Rt = new float[16]; // Contains rigid transform matrix
//         Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;
//         align2tsdf(
//             tsdfA.data.data(), tsdfA.dim.x, tsdfA.dim.y, tsdfA.dim.z, tsdfA.origin.x, tsdfA.origin.y, tsdfA.origin.z,
//             tsdfB.data.data(), tsdfB.dim.x, tsdfB.dim.y, tsdfB.dim.z, tsdfB.origin.x, tsdfB.origin.y, tsdfB.origin.z,
//             voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

//         ///////////////////////////////////////////////////////////////////

//         // TODO: this is redundant with align2tsdf
//         Result result;
//         result.keypointsA = tsdfA.makeKeypoints();
//         result.keypointsB = tsdfB.makeKeypoints();

//         result.transformBToA = mat4f::identity();
//         for (int i = 0; i < 4; i++)
//             for (int j = 0; j < 3; j++)
//                 result.transformBToA(j, i) = Rt[j * 4 + i];
        
//         pc2tsdf::UniformAccelerator acceleratorA(result.keypointsA, maxKeypointMatchDist);
        
//         //transform B's keypoints into A's
//         std::vector<vec3f> keypointsBtransformed = result.keypointsB;
//         for (auto &bPt : keypointsBtransformed)
//         {
//             const vec3f bPtInA = result.transformBToA * bPt;
//             const auto closestPt = acceleratorA.findClosestPoint(bPtInA);
//             const float dist = vec3f::dist(bPtInA, closestPt.first);
//             if (dist <= maxKeypointMatchDist)
//             {
//                 KeypointMatch match;
//                 match.posA = closestPt.first;
//                 match.posB = bPt;
//                 match.alignmentError = dist;
//                 result.matches.push_back(match);
//             }
//         }

//         result.matchFound = result.matches.size() > 0;

//         std::cout << "Keypoint matches found: " << result.matches.size() << std::endl;

//         ///////////////////////////////////////////////////////////////////
        
//         float* final_Rt = new float[16];
//         final_Rt = Rt;

//         // DISABLE ICP FOR NOW (too slow)
//         bool use_matlab_icp = false;
//         if (use_matlab_icp) {
//           tic();
//           // Save point clouds to files for matlab to read
//           auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
//           FILE *fp = fopen("TMPpointcloud1.txt", "w");
//           for (int i = 0; i < cloud1.m_points.size(); i++)
//             fprintf(fp, "%f %f %f\n", cloud1.m_points[i].x, cloud1.m_points[i].y, cloud1.m_points[i].z);
//           fclose(fp);
//           auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);
//           fp = fopen("TMPpointcloud2.txt", "w");
//           for (int i = 0; i < cloud2.m_points.size(); i++) {
//             vec3f tmp_point;
//             tmp_point.x = Rt[0] * cloud2.m_points[i].x + Rt[1] * cloud2.m_points[i].y + Rt[2] * cloud2.m_points[i].z + Rt[3];
//             tmp_point.y = Rt[4] * cloud2.m_points[i].x + Rt[5] * cloud2.m_points[i].y + Rt[6] * cloud2.m_points[i].z + Rt[7];
//             tmp_point.z = Rt[8] * cloud2.m_points[i].x + Rt[9] * cloud2.m_points[i].y + Rt[10] * cloud2.m_points[i].z + Rt[11];
//             fprintf(fp, "%f %f %f\n", tmp_point.x, tmp_point.y, tmp_point.z);
//           }
//           fclose(fp);

//           // Run matlab ICP
//           sys_command("cd matlab; matlab -nojvm < main.m >/dev/null; cd ..");
//           float *icp_Rt = new float[16];
//           int iret;
//           fp = fopen("TMPicpRt.txt", "r");
//           for (int i = 0; i < 16; i++) {
//             iret = fscanf(fp, "%f", &icp_Rt[i]);
//           }
//           fclose(fp);

//           // Apply ICP Rt to current Rt
//           mulMatrix(icp_Rt, Rt, final_Rt);

//           delete [] icp_Rt;
//           sys_command("rm TMPpointcloud1.txt");
//           sys_command("rm TMPpointcloud2.txt");
//           sys_command("rm TMPicpRt.txt");

//           std::cout << "Using ICP to re-adjust rigid transform. ";
//           toc();
//         }

//         const bool debugDump = true;
//         if (debugDump) {
//             ///////////////////////////////////////////////////////////////////
//             // DEBUG: save point aligned point clouds
//             tic();

//             auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
//             auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);

//             // Rotate B points into A using final_Rt
//             for (int i = 0; i < cloud2.m_points.size(); i++) {
//                 vec3f tmp_point;
//                 tmp_point.x = final_Rt[0] * cloud2.m_points[i].x + final_Rt[1] * cloud2.m_points[i].y + final_Rt[2] * cloud2.m_points[i].z + final_Rt[3];
//                 tmp_point.y = final_Rt[4] * cloud2.m_points[i].x + final_Rt[5] * cloud2.m_points[i].y + final_Rt[6] * cloud2.m_points[i].z + final_Rt[7];
//                 tmp_point.z = final_Rt[8] * cloud2.m_points[i].x + final_Rt[9] * cloud2.m_points[i].y + final_Rt[10] * cloud2.m_points[i].z + final_Rt[11];
//                 cloud2.m_points[i] = tmp_point;
//             }

//             // Make point clouds colorful
//             ml::vec4f color1;
//             for (int i = 0; i < 3; i++)
//                 color1[i] = gen_random_float(0.0, 1.0);
//             for (int i = 0; i < cloud1.m_points.size(); i++)
//                 cloud1.m_colors[i] = color1;
//             ml::vec4f color2;
//             for (int i = 0; i < 3; i++)
//                 color2[i] = gen_random_float(0.0, 1.0);
//             for (int i = 0; i < cloud2.m_points.size(); i++)
//                 cloud2.m_colors[i] = color2;

//             // Save point clouds to file
//             std::string pcfile1 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndA) + ".ply";
//             PointCloudIOf::saveToFile(pcfile1, cloud1);
//             std::string pcfile2 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndB) + ".ply";
//             PointCloudIOf::saveToFile(pcfile2, cloud2);

//             std::cout << "Saving point cloud visualizations. ";
//             toc();
//         }


























  // std::cout << sensorData.m_depthShift << std::endl;

  // // unsigned short* tmp_depth = sensorData.m_frames[0].decompressDepthAlloc();
  // unsigned short* tmp_depth = sensorData.m_frames[0].decompressDepthAlloc_stb();
  // // ml::vec3uc* tmp_color = sensorData.m_frames[0].decompressColorAlloc();

  // std::cout << ((float)tmp_depth[0])/sensorData.m_depthShift << std::endl;
  // float max = 0; float min = 100000;
  // for (int j = 600; j < 650; j++) {
  //   unsigned short* tmp_depth = sensorData.m_frames[j].decompressDepthAlloc_stb();
  //   for (int i = 0; i < sensorData.m_depthWidth*sensorData.m_depthHeight; i++) {
  //     if (((float)tmp_depth[i])/sensorData.m_depthShift > max) {
  //       max = ((float)tmp_depth[i])/sensorData.m_depthShift;
  //     }
  //     if (((float)tmp_depth[i])/sensorData.m_depthShift < min) {
  //       min = ((float)tmp_depth[i])/sensorData.m_depthShift;
  //     }
  //     // std::cout << (float)sensorData.m_frames[j].m_depthCompressed[i]/sensorData.m_depthShift << std::endl;
  //   }
  // }
  // std::cout << max << std::endl;
  // std::cout << min << std::endl;

  // for (int i = 0; i < 16; i++)
  //   std::cout << sensorData.m_frames[600].m_frameToWorld.matrix[i] << std::endl;

  // std::ofstream ofs ("test.jpg", std::ofstream::binary);
  // ofs.write ((const char*)sensorData.m_frames[600].m_colorCompressed, sensorData.m_frames[0].m_colorSizeBytes);
  // ofs.close();
  // // std::cout << (int)sensorData.m_frames[0].m_colorCompressed[0] << std::endl;











  // generateTestingData();





  ///////////////////////////////////////////////////////////////////

  // std::string scene1_dir = "/data/andyz/kinfu/sun3d/scene0_30";
  // std::string scene2_dir = "/data/andyz/kinfu/sun3d/scene630_660";

  // float *scene1_tsdf = new float[512 * 512 * 1024];
  // checkout_tsdf(scene1_dir, scene1_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  // std::vector<keypoint> keypoints1;
  // checkout_keypts(scene1_dir, keypoints1);

  // float tsdf_threshold = 0.2;
  // int x_dim = 512; int y_dim = 512; int z_dim = 1024;
  // std::string ply_filename1 = "test_scene0_30.ply";
  // float *ext_mat = new float[16];
  // checkout_ext("scene0_30", ext_mat);
  // tsdf2ply(ply_filename1, scene1_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);

  // float *scene2_tsdf = new float[512 * 512 * 1024];
  // checkout_tsdf(scene2_dir, scene2_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  // std::vector<keypoint> keypoints2;
  // checkout_keypts(scene2_dir, keypoints2);

  // // Save keypoint point clouds
  // std::string keypoints_dir = "keypoints";
  // sys_command("mkdir -p " + keypoints_dir);
  // for (int i = 0; i < keypoints1.size(); i++) {
  //   std::string tmp_filename = keypoints_dir + "/scene1_" + std::to_string((int)keypoints1[i].x) + "_" + std::to_string((int)keypoints1[i].y) + "_" + std::to_string((int)keypoints1[i].z) + ".ply";
  //   tsdfpt2ply(tmp_filename, scene1_tsdf, 0.2, keypoints1[i], 512, 512, 1024);
  // }
  // for (int i = 0; i < keypoints2.size(); i++) {
  //   std::string tmp_filename = keypoints_dir + "/scene2_" + std::to_string((int)keypoints2[i].x) + "_" + std::to_string((int)keypoints2[i].y) + "_" + std::to_string((int)keypoints2[i].z) + ".ply";
  //   tsdfpt2ply(tmp_filename, scene2_tsdf, 0.2, keypoints2[i], 512, 512, 1024);
  // }

  // // Test comparing one pair of keypoints
  // std::vector<keypoint> tmpkeypoints1;
  // tmpkeypoints1.push_back(keypoints1[122]);
  // tmpkeypoints1.push_back(keypoints2[43]);
  // std::vector<keypoint> tmpkeypoints2;
  // tmpkeypoints2.push_back(keypoints1[122]);
  // tmpkeypoints2.push_back(keypoints2[43]);





  // tripleD(scene1_dir, scene2_dir, keypoints1, keypoints2);










  ///////////////////////////////////////////////////////////////////

  // // Init extrinsic matrix to identity
  // float *ext_mat = new float[16];
  // for (int i = 0; i < 16; i++)
  //   ext_mat[i] = 0.0f;
  // ext_mat[0] = 1.0f;
  // ext_mat[5] = 1.0f;
  // ext_mat[10] = 1.0f;
  // ext_mat[15] = 1.0f;

  // for (int i = 50; i < 2869 - 50; i = i + 50) { // 1868 - 30

  //   // // Sun3D scene directories
  //   // std::string scene1_dir = "/data/andyz/kinfu/sun3d/scene" + std::to_string(i - 30) + "_" + std::to_string(i);
  //   // std::string scene2_dir = "/data/andyz/kinfu/sun3d/scene" + std::to_string(i) + "_" + std::to_string(i + 30);

  //   // // Point cloud filenames
  //   // std::string ply_filename1 = "test_scene" + std::to_string(i - 30) + "_" + std::to_string(i) + ".ply";
  //   // std::string ply_filename2 = "test_scene" + std::to_string(i) + "_" + std::to_string(i + 30) + ".ply";

  //   // ICLNUIM scene directories
  //   std::string scene1_dir = "/data/andyz/kinfu/iclnuim/livingroom1scene" + std::to_string(i - 50) + "_" + std::to_string(i);
  //   std::string scene2_dir = "/data/andyz/kinfu/iclnuim/livingroom1scene" + std::to_string(i) + "_" + std::to_string(i + 50);

  //   // Point cloud filenames
  //   std::string ply_filename1 = "test_scene" + std::to_string(i - 50) + "_" + std::to_string(i) + ".ply";
  //   std::string ply_filename2 = "test_scene" + std::to_string(i) + "_" + std::to_string(i + 50) + ".ply";

  //   // Load first scene TSDF and its keypoints
  //   float *scene1_tsdf = new float[512 * 512 * 1024];
  //   checkout_tsdf(scene1_dir, scene1_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  //   std::vector<keypoint> keypoints1;
  //   checkout_keypts(scene1_dir, keypoints1);

  //   // Load second scene TSDF and its keypoints
  //   float *scene2_tsdf = new float[512 * 512 * 1024];
  //   checkout_tsdf(scene2_dir, scene2_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  //   std::vector<keypoint> keypoints2;
  //   checkout_keypts(scene2_dir, keypoints2);

  //   // Init point cloud file params
  //   float tsdf_threshold = 0.2;
  //   int x_dim = 512; int y_dim = 512; int z_dim = 1024;

  //   // Save first scene as point cloud
  //   keypoint pc_color;
  //   if (i == 50) {
  //     pc_color.x = gen_random_float(0.0f, 255.0f);
  //     pc_color.y = gen_random_float(0.0f, 255.0f);
  //     pc_color.z = gen_random_float(0.0f, 255.0f);
  //     tsdf2ply(ply_filename1, scene1_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim, false, pc_color);
  //   }

  //   // Compute extrinsic to warp second point cloud
  //   // tripleD(scene1_dir, scene2_dir, keypoints1, keypoints2);
  //   tripleD_modularized(scene1_dir, scene2_dir, keypoints1, keypoints2);


  //   sys_command("cp " + scene1_dir + "_pts.txt TMPscene1_pts.txt");
  //   sys_command("cp " + scene2_dir + "_pts.txt TMPscene2_pts.txt");
  //   sys_command("cd matlab; matlab -nojvm < main.m; cd ..");
  //   // sys_command("rm TMPscene1_pts.txt");
  //   // sys_command("rm TMPscene2_pts.txt");
  //   float *tmp_ext_mat = new float[16];
  //   int iret;
  //   FILE *fp = fopen("TMPrigidtransform.txt", "r");
  //   for (int i = 0; i < 16; i++) {
  //     iret = fscanf(fp, "%f", &tmp_ext_mat[i]);
  //     std::cout << tmp_ext_mat[i] << std::endl;
  //   }
  //   fclose(fp);
  //   sys_command("rm TMPrigidtransform.txt");
  //   float *curr_ext_mat = new float[16];
  //   mulMatrix(ext_mat, tmp_ext_mat, curr_ext_mat);
  //   for (int i = 0; i < 16; i++)
  //     ext_mat[i] = curr_ext_mat[i];

  //   // Save second scene as point cloud
  //   pc_color.x = gen_random_float(0.0f, 255.0f);
  //   pc_color.y = gen_random_float(0.0f, 255.0f);
  //   pc_color.z = gen_random_float(0.0f, 255.0f);
  //   tsdf2ply(ply_filename2, scene2_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim, true, pc_color);

  //   // Clear memory
  //   delete [] scene1_tsdf;
  //   delete [] scene2_tsdf;
  //   delete [] tmp_ext_mat;
  //   delete [] curr_ext_mat;

  // }

  ///////////////////////////////////////////////////////////////////














  // std::string train_dir = "/data/04/andyz/kinfu/train/fire_seq03/";
  // std::string data_dir = "/data/04/andyz/kinfu/data/sevenscenes/fire/seq-03/";
  // generate_data(train_dir, data_dir);

















  // 0, yellow thing
  // 20, clothing
  // Choose random point in training data set
  // Choose random match
  // Choose random non-match
  // Tip: random flip

  // time_t tstart, tend;
  // tstart = time(0);
  // for (int i = 0; i < 256; i++) {
  //   float *volume_o_tsdf = new float[31 * 31 * 31];
  //   float *volume_m_tsdf = new float[31 * 31 * 31];
  //   float *volume_nm_tsdf = new float[31 * 31 * 31];
  //   std::string data_directory = "/data/andyz/kinfu/train";
  //   std::cout << std::endl;
  //   create_match_nonmatch(data_directory, volume_o_tsdf, volume_m_tsdf, volume_nm_tsdf, i);
  //   delete [] volume_o_tsdf;
  //   delete [] volume_m_tsdf;
  //   delete [] volume_nm_tsdf;
  // }
  // tend = time(0);
  // std::cout << "One batch took " << difftime(tend, tstart) << " second(s)." << std::endl;

  // float *volume_o_tsdf = new float[31 * 31 * 31];
  // float *volume_m_tsdf = new float[31 * 31 * 31];
  // float *volume_nm_tsdf = new float[31 * 31 * 31];
  // std::string data_directory = "/data/andyz/kinfu/train";
  // std::cout << std::endl;
  // create_match_nonmatch(data_directory, volume_o_tsdf, volume_m_tsdf, volume_nm_tsdf, 1);




  // for (int i = 0; i < 10; i++)
  //   cout << volume_o_tsdf[i] << endl;

  // for (int i = 0; i < 10; i++)
  //   cout << volume_m_tsdf[i] << endl;

  // for (int i = 0; i < 10; i++)
  //   cout << volume_nm_tsdf[i] << endl;







  // cout << scene_dir_nm << endl;

  // string tsdf_match_string = ".tsdf";
  // get_files_in_directory(train_dir, scene_names, tsdf_match_string);










  // // Pick random scenes
  // int rand_scene_i_o = (int) floor(gen_random_float(0.0, scene_names_o.size()));
  // int rand_scene_i_m = (int) floor(gen_random_float(0.0, scene_names_o.size()));
  // string scene_dir = train_dir + "/" + scene_names[rand_scene_i];


  // cout << endl;

  // for (int i = 0; i < scene_names_o.size(); i++) {
  //   cout << scene_names_o[i] << endl;
  // }

  // cout << endl;

  // for (int i = 0; i < scene_names_nm.size(); i++) {
  //   cout << scene_names_nm[i] << endl;
  // }













  // float *scene_tsdf = new float[512 * 512 * 1024];
  // checkout_tsdf("/data/andyz/kinfu/synth/synth_00b124b28874d724.sss/random_0", scene_tsdf, 0, 512-1, 0, 512-1, 0, 1024-1);
  // float tsdf_threshold = 0.2;
  // int x_dim = 512; int y_dim = 512; int z_dim = 1024;
  // std::string ply_filename = "test_random_0.ply";
  // float *ext_mat = new float[16];
  // checkout_ext("/data/andyz/kinfu/synth/synth_00b124b28874d724.sss/random_0", ext_mat);
  // tsdf2ply(ply_filename, scene_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);
  // free(scene_tsdf);














  // keypoint keypt_world;


  // // Convert grid coordinates to world coordinates
  // float sx = ((float)keypt_grid.x+1)*0.01-512*0.01/2;
  // float sy = ((float)keypt_grid.y+1)*0.01-512*0.01/2;
  // float sz = ((float)keypt_grid.z+1)*0.01-0.5;
  // keypt_world.x = ext_mat[0]*sx + ext_mat[1]*sy + ext_mat[2]*sz;
  // keypt_world.y = ext_mat[4]*sx + ext_mat[5]*sy + ext_mat[6]*sz;
  // keypt_world.z = ext_mat[8]*sx + ext_mat[9]*sy + ext_mat[10]*sz;
  // keypt_world.x = keypt_world.x + ext_mat[3];
  // keypt_world.y = keypt_world.y + ext_mat[7];
  // keypt_world.z = keypt_world.z + ext_mat[11];

  // cout << keypt_world.x << " " << keypt_world.y << " " << keypt_world.z << endl;

  // vector<string> scene_names;
  // scene_names.push_back("frame0-30_base0");
  // scene_names.push_back("frame30-60_base30");
  // scene_names.push_back("frame150-180_base150");
  // scene_names.push_back("frame210-240_base210");
  // scene_names.push_back("frame960-990_base960");

  // for (int i = 0; i <= 960; i = i+30) {
  //   string scene_name = "frame" + std::to_string(i) + "-" + std::to_string(i+30) + "_base" + std::to_string(i);

  //   // Convert world coordinates to grid coordinates
  //   checkout_ext(scene_name, ext_mat);
  //   float *ext_mat_inv = new float[16];
  //   invMatrix(ext_mat,ext_mat_inv);
  //   keypt_grid.x = ext_mat_inv[0]*keypt_world.x + ext_mat_inv[1]*keypt_world.y + ext_mat_inv[2]*keypt_world.z;
  //   keypt_grid.y = ext_mat_inv[4]*keypt_world.x + ext_mat_inv[5]*keypt_world.y + ext_mat_inv[6]*keypt_world.z;
  //   keypt_grid.z = ext_mat_inv[8]*keypt_world.x + ext_mat_inv[9]*keypt_world.y + ext_mat_inv[10]*keypt_world.z;
  //   keypt_grid.x = keypt_grid.x + ext_mat_inv[3];
  //   keypt_grid.y = keypt_grid.y + ext_mat_inv[7];
  //   keypt_grid.z = keypt_grid.z + ext_mat_inv[11];
  //   keypt_grid.x = round(((keypt_grid.x+512*0.01/2)/0.01)-1);
  //   keypt_grid.y = round(((keypt_grid.y+512*0.01/2)/0.01)-1);
  //   keypt_grid.z = round(((keypt_grid.z+0.5)/0.01)-1);

  //   bool keypt_is_valid = false;
  //   checkout_keypts(scene_name, keypoints);

  //   for (int j = 0; j < keypoints.size(); j++)
  //     if (((keypoints[j].x-keypt_grid.x)*(keypoints[j].x-keypt_grid.x)+(keypoints[j].y-keypt_grid.y)*(keypoints[j].y-keypt_grid.y)+(keypoints[j].z-keypt_grid.z)*(keypoints[j].z-keypt_grid.z))<25)
  //       keypt_is_valid = true;

  //   if (keypt_is_valid) {
  //     checkout_data(scene_name, keypt_grid);
  //     cout << "Finished: " << scene_name << endl;
  //   }

  // }

  return 0;
}


// if (view_occupancy < 0.7 || curr_frame == total_files-1) {

//   // Save curr volume to file
//   string scene_name = "frame" + to_string(first_frame) + "-" + to_string(curr_frame-1) + "_" + "base" + to_string(base_frame) + ".ply";
//   save_volume_to_ply(scene_name);

//   // Check for loop closure and set a base frame
//   base_frame = curr_frame;
//   float base_frame_intersection = 0;
//   for (int i = 0; i < base_frames.size(); i++) {
//     float cam_R[9] = {0};
//     float cam_t[3] = {0};
//     float range_grid[6] = {0};
//     compute_frustum_bounds(extrinsic_poses, base_frames[i], curr_frame, cam_R, cam_t, range_grid);
//     float view_occupancy = (range_grid[1]-range_grid[0])*(range_grid[3]-range_grid[2])*(range_grid[5]-range_grid[4])/(512*512*1024);
//     if (view_occupancy > max(0.7f, base_frame_intersection)) {
//       base_frame = base_frames[i];
//       base_frame_intersection = view_occupancy;
//     }
//   }
//   base_frames.push_back(base_frame);

//   // Init new volume
//   first_frame = curr_frame;
//   curr_frame = curr_frame-1;
//   cout << "Creating new volume." << endl;
//   memset(voxel_volume.weight, 0, sizeof(float)*512*512*1024);
//   for (int i = 0; i < 512*512*1024; i++)
//     voxel_volume.tsdf[i] = 1.0f;

// }


// #define MAX_THREADS (2048*2048)
// #define THREADS_PER_BLOCK 512

// __global__ void saxpy(int n, float a, float *x, float *y)
// {
//   int i = blockIdx.x*blockDim.x + threadIdx.x;
//   if (i < n) y[i] = a*x[i] + y[i];
// }

// int N = 1<<20;
// float *x, *y, *d_x, *d_y;
// x = (float*)malloc(N*sizeof(float));
// y = (float*)malloc(N*sizeof(float));

// cudaMalloc(&d_x, N*sizeof(float));
// cudaMalloc(&d_y, N*sizeof(float));

// for (int i = 0; i < N; i++) {
//   x[i] = 1.0f;
//   y[i] = 2.0f;
// }

// cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
// cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

// // Perform SAXPY on 1M elements
// saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

// cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

// float maxError = 0.0f;
// for (int i = 0; i < N; i++)
//   maxError = max(maxError, abs(y[i]-4.0f));
// printf("Max error: %fn", maxError);




// //////////////////////////////////////////////////////////////////////////////////////////////////////

// // Init cuda vars
// // uchar *g_image_data;
// // ushort *g_depth_data;
// // float *g_range_grid, *g_cam_R, *g_cam_t;
// // float *g_tsdf, *g_weight;
// // cudaMalloc(&g_image_data, kImageRows * kImageCols * kImageChannels * sizeof(uchar));
// // cudaMalloc(&g_depth_data, kImageRows * kImageCols * sizeof(ushort));
// // cudaMalloc(&g_range_grid, 6*sizeof(float));
// // cudaMalloc(&g_cam_R, 9*sizeof(float));
// // cudaMalloc(&g_cam_t, 3*sizeof(float));
// // cudaMalloc(&g_tsdf, 512*512*1024*sizeof(float));
// // cudaMalloc(&g_weight, 512*512*1024*sizeof(float));

// // Copy volume data to GPU
// // cudaMemcpy(g_image_data, image_data, kImageRows * kImageCols * kImageChannels * sizeof(uchar), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_depth_data, depth_data, kImageRows * kImageCols * sizeof(ushort), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_range_grid, range_grid, 6*sizeof(float), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_cam_R, cam_R, 9*sizeof(float), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_cam_t, cam_t, 3*sizeof(float), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_tsdf, voxel_volume.tsdf, 512*512*1024*sizeof(float), cudaMemcpyHostToDevice);
// // cudaMemcpy(g_weight, voxel_volume.weight, 512*512*1024*sizeof(float), cudaMemcpyHostToDevice);


// saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

// for (int i = 0; i < 5; i++) {
//   cout << voxel_volume.tsdf[i] << endl;
// }

// Perform SAXPY on 1M elements
// integrate<<<(512*512*1024+(THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(g_tsdf, g_weight, g_image_data, g_depth_data, g_range_grid, g_cam_R, g_cam_t);

// Copy volume data back to CPU
// cudaMemcpy(voxel_volume.tsdf, g_tsdf, 512*512*1024*sizeof(float), cudaMemcpyDeviceToHost);
// cudaMemcpy(voxel_volume.weight, g_weight, 512*512*1024*sizeof(float), cudaMemcpyDeviceToHost);
// for (int i = 0; i < 5; i++)
//   cout << voxel_volume.tsdf[i] << endl;




// cudaFree(g_image_data);
// cudaFree(g_depth_data);
// cudaFree(g_range_grid);
// cudaFree(g_cam_R);
// cudaFree(g_cam_t);
// cudaFree(g_tsdf);
// cudaFree(g_weight);




////////////////////////////////////////////////////////////////

// const std::string sensor_file = "/data/andyz/kinfu/data/relocDataset/APTA/kitchen/akitchen.sens";
//   std::cout << "loading sensor file ... ";
//   ml::SensorData sensorData(sensor_file);
//   std::cout << "DONE!" << std::endl;
//   std::cout << sensorData << std::endl;

//   if (true) {
//     // const unsigned int numBuckets = 10;
//     // float minValue = 0.0f;
//     // float maxValue = 3.0f;
//     // unsigned int historgram[numBuckets];
//     // for (unsigned int i = 0; i < numBuckets; i++) historgram[i] = 0;
//     // float step = (maxValue - minValue) / (float)numBuckets;

//     std::string src = "/data/andyz/kinfu/data/relocDataset/APTA/kitchen/akitchen.sens";
//     ml::SensorData srcData(src);
//     unsigned short* depth = srcData.m_frames[0].decompressDepthAlloc();
//     // unsigned int sum = 0;
//     for (unsigned int i = 0; i < srcData.m_depthWidth * srcData.m_depthHeight; i++) {
//       unsigned short d = depth[i];
//       if (d) {
//         float df = (float)d / srcData.m_depthShift;
//         // float cf = df / step;
//         // std::cout << df << std::endl;
//         // std::cout << df << " " << cf << std::endl << std::endl;
//         // unsigned int entry = std::min(std::max(std::round(cf), 0.0f), (float)numBuckets - 1);
//         // historgram[entry]++;
//         // sum++;
//         // std::cout <<  << std::endl;
//       }
//     }
//     // for (unsigned int i = 0; i < numBuckets; i++) {
//     //   float perc = (float)historgram[i] / (float)sum; perc *= 100.0f;
//     //   std::cout << i*step << "m to " << (i + 1)*step << "m = " << perc << "%" << std::endl;
//     // }
//     std::free(depth);
//   }


/////////////////////////////////////////////////////////////////////

// std::string sensor_file;
// std::string local_dir;
// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTA/kitchen/akitchen.sens";
// local_dir = "/data/andyz/kinfu/reloc/apta_kitchen/";
// generate_data_reloc(sensor_file, local_dir, 0, 356);
// generate_data_reloc(sensor_file, local_dir, 357, 714);
// generate_data_reloc(sensor_file, local_dir, 715, 1100);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTA/living/aliving.sens";
// local_dir = "/data/andyz/kinfu/reloc/apta_living/";
// generate_data_reloc(sensor_file, local_dir, 0, 492);
// generate_data_reloc(sensor_file, local_dir, 493, 1021);
// generate_data_reloc(sensor_file, local_dir, 1022, 1527);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTM/bed/bed.sens";
// local_dir = "/data/andyz/kinfu/reloc/aptm_bed/";
// generate_data_reloc(sensor_file, local_dir, 0, 243);
// generate_data_reloc(sensor_file, local_dir, 244, 529);
// generate_data_reloc(sensor_file, local_dir, 530, 707);
// generate_data_reloc(sensor_file, local_dir, 708, 903);
// generate_data_reloc(sensor_file, local_dir, 904, 1137);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTM/kitchen/kitchen.sens";
// local_dir = "/data/andyz/kinfu/reloc/aptm_kitchen/";
// generate_data_reloc(sensor_file, local_dir, 0, 229);
// generate_data_reloc(sensor_file, local_dir, 230, 385);
// generate_data_reloc(sensor_file, local_dir, 386, 590);
// generate_data_reloc(sensor_file, local_dir, 591, 819);
// generate_data_reloc(sensor_file, local_dir, 820, 1011);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTM/living/mliving.sens";
// local_dir = "/data/andyz/kinfu/reloc/aptm_living/";
// generate_data_reloc(sensor_file, local_dir, 0, 358);
// generate_data_reloc(sensor_file, local_dir, 359, 744);
// generate_data_reloc(sensor_file, local_dir, 745, 1089);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/APTM/luke/luke.sens";
// local_dir = "/data/andyz/kinfu/reloc/aptm_luke/";
// generate_data_reloc(sensor_file, local_dir, 0, 623);
// generate_data_reloc(sensor_file, local_dir, 624, 1216);
// generate_data_reloc(sensor_file, local_dir, 1217, 1605);
// generate_data_reloc(sensor_file, local_dir, 1606, 1993);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/FLOOR5/5a/5a.sens";
// local_dir = "/data/andyz/kinfu/reloc/floor5_5a/";
// generate_data_reloc(sensor_file, local_dir, 0, 496);
// generate_data_reloc(sensor_file, local_dir, 497, 1030);
// generate_data_reloc(sensor_file, local_dir, 1031, 1497);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/FLOOR5/5b/5b.sens";
// local_dir = "/data/andyz/kinfu/reloc/floor5_5b/";
// generate_data_reloc(sensor_file, local_dir, 0, 414);
// generate_data_reloc(sensor_file, local_dir, 415, 932);
// generate_data_reloc(sensor_file, local_dir, 933, 1391);
// generate_data_reloc(sensor_file, local_dir, 1392, 1855);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/GATES/gates362/gates362.sens";
// local_dir = "/data/andyz/kinfu/reloc/gates_gates362/";
// generate_data_reloc(sensor_file, local_dir, 0, 385);
// generate_data_reloc(sensor_file, local_dir, 386, 1326);
// generate_data_reloc(sensor_file, local_dir, 1327, 2278);
// generate_data_reloc(sensor_file, local_dir, 2279, 3316);
// generate_data_reloc(sensor_file, local_dir, 3317, 3925);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/GATES/lounge/lounge.sens";
// local_dir = "/data/andyz/kinfu/reloc/gates_lounge/";
// generate_data_reloc(sensor_file, local_dir, 0, 326);
// generate_data_reloc(sensor_file, local_dir, 327, 484);
// generate_data_reloc(sensor_file, local_dir, 485, 709);
// generate_data_reloc(sensor_file, local_dir, 710, 1049);
// generate_data_reloc(sensor_file, local_dir, 1050, 1259);

// sensor_file = "/data/andyz/kinfu/data/relocDataset/GATES/manolis/manolis.sens";
// local_dir = "/data/andyz/kinfu/reloc/gates_manolis/";
// generate_data_reloc(sensor_file, local_dir, 0, 806);
// generate_data_reloc(sensor_file, local_dir, 807, 1532);
// generate_data_reloc(sensor_file, local_dir, 1533, 2429);

/////////////////////////////////////////////////////////////////////