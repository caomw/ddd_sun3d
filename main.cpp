#include <kinfu.hpp>
#include <fragments/io.h>

inline bool exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// New format:
// 3 ints (X,Y,Z volume size), followed by float array of TSDF
void convert_tsdf_old_to_new(const std::string &src, const std::string &dst) {
  
  // Read TSDF from old format
  float *tsdf = new float[512 * 512 * 1024];
  checkout_tsdf(src, tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
  
  // Find dimensions of TSDF volume
  int x_min = 511, x_max = 0, y_min = 511, y_max = 0, z_min = 1023, z_max = 0; // inclusive bounds
  for (int z = 0; z < 1024; z++)
    for (int y = 0; y < 512; y++)
      for (int x = 0; x < 512; x++) {
        int i = z*512*512 + y*512 + x;
        if (tsdf[i] < 1) {
          x_max = std::max(x, x_max);
          y_max = std::max(y, y_max);
          z_max = std::max(z, z_max);
        }
      }
  for (int z = 1023; z >= 0; z--)
    for (int y = 511; y >= 0; y--)
      for (int x = 511; x >= 0; x--) {
        int i = z*512*512 + y*512 + x;
        if (tsdf[i] < 1) {
          x_min = std::min(x, x_min);
          y_min = std::min(y, y_min);
          z_min = std::min(z, z_min);
        }
      }
  int x_dim = x_max - x_min + 1;
  int y_dim = y_max - y_min + 1;
  int z_dim = z_max - z_min + 1;

  // // Debug
  // std::cout << x_min << std::endl;
  // std::cout << x_max << std::endl;
  // std::cout << y_min << std::endl;
  // std::cout << y_max << std::endl;
  // std::cout << z_min << std::endl;
  // std::cout << z_max << std::endl;

  // Create new smaller TSDF volume
  float *new_tsdf = new float[x_dim * y_dim * z_dim];
  for (int z = 0; z < z_dim; z++)
    for (int y = 0; y < y_dim; y++)
      for (int x = 0; x < x_dim; x++)
        new_tsdf[z*y_dim*x_dim + y*x_dim + x] = tsdf[(z+z_min)*512*512 + (y+y_min)*512 + (x+x_min)];

  write_fragment_tsdf(dst, new_tsdf, x_dim, y_dim, z_dim); // function found in fragments/io.h

  delete [] tsdf;
  delete [] new_tsdf;

}

int main(int argc, char **argv) {



  // // CODE TO CONVERT OLD TSDF FORMAT TO NEW TSDF FORMAT
  // convert_tsdf_old_to_new("/data/andyz/kinfu/sun3d/mit_76_studyroom/76-1studyroom2/scene960_1010", "fragments/test_new.tsdf");

  // // CODE TO READ NEW TSDF FORMAT AND MAKE POINT CLOUD
  // demo_read_fragment_tsdf("fragments/test_new.tsdf");


  // CHANGE THESE IF YOU NEED TO
  bool ddd_verbose = true;
  const float k_match_score_thresh = 0.07f;
  const float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
  const float max_ransac_iter = 1000000;
  const float ransac_inlier_thresh = 8.0f; // distance in grid coordinates, default: 4 voxels


  std::string sequence_name =  argv[1];
  int fuse_frame_start_idx_i = atoi(argv[2]);
  int fuse_frame_start_idx_j = atoi(argv[3]); 
  int num_frames_per_frag = 50;
  /*
  std::string sequence_name = "mit_32_d507/d507_2/";
  int fuse_frame_start_idx_i = 1330;
  int fuse_frame_start_idx_j = 3000;//1630 
  int num_frames_per_frag = 50;
  */
  /*
  std::string sequence_name = "mit_76_studyroom/76-1studyroom2/";
  int fuse_frame_start_idx_i = 960;
  int fuse_frame_start_idx_j = 1630;//1630 
  int num_frames_per_frag = 50;
  */
  /*
  std::string sequence_name = "mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/";
  int fuse_frame_start_idx_i = 300;
  int fuse_frame_start_idx_j = 1370;//1630 
  int num_frames_per_frag = 50;
  */

  
  std::string frag_saveto_dir;
  std::string sun3d_data_load_dir;

  sun3d_data_load_dir = "/data/andyz/kinfu/data/sun3d/";

  //Fuse first fragment (CAN COMMENT OUT THIS SECTION IF YOU ALREADY FUSED)
  frag_saveto_dir = "/data/andyz/kinfu/sun3d/"+sequence_name;
  int result = system(("mkdir -p "+ frag_saveto_dir).c_str());
  
  std::string scene_ply_name = frag_saveto_dir + "scene" +  std::to_string(fuse_frame_start_idx_i) + "_" + std::to_string(fuse_frame_start_idx_i + num_frames_per_frag) + ".ply";
  if (!exists(scene_ply_name)){
     generate_data_sun3d(sequence_name, frag_saveto_dir, sun3d_data_load_dir, fuse_frame_start_idx_i, fuse_frame_start_idx_i+num_frames_per_frag, num_frames_per_frag);
  }
  else{
    std::cout<<"file exist, skip fusing: " << scene_ply_name << std::endl;
  }

  //Fuse second fragment (CAN COMMENT OUT THIS SECTION IF YOU ALREADY FUSED)
  scene_ply_name = frag_saveto_dir + "scene" +  std::to_string(fuse_frame_start_idx_j) + "_" + std::to_string(fuse_frame_start_idx_j + num_frames_per_frag) + ".ply";
  if (!exists(scene_ply_name)){
    generate_data_sun3d(sequence_name, frag_saveto_dir, sun3d_data_load_dir, fuse_frame_start_idx_j,  fuse_frame_start_idx_j+num_frames_per_frag, num_frames_per_frag);
  }
  else{
    std::cout<<"file exist, skip fusing: " << scene_ply_name << std::endl;
  }

  // CHANGE THESE IF YOU NEED TO
  std::string scene1_dir = "/data/andyz/kinfu/sun3d/"+ sequence_name + "/scene" + std::to_string(fuse_frame_start_idx_i) + "_" + std::to_string(fuse_frame_start_idx_i + num_frames_per_frag);
  std::string scene2_dir = "/data/andyz/kinfu/sun3d/"+ sequence_name + "/scene" + std::to_string(fuse_frame_start_idx_j) + "_" + std::to_string(fuse_frame_start_idx_j + num_frames_per_frag);




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

  // Take absolute value of TSDF volume
  for (int i = 0; i < 512 * 512 * 1024; i++)
    scene_tsdf1[i] = std::abs(scene_tsdf1[i]);

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

  // Take absolute value of TSDF volume
  for (int i = 0; i < 512 * 512 * 1024; i++)
    scene_tsdf2[i] = std::abs(scene_tsdf2[i]);

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
  float tsdf_threshold = 0.03f;
  int num_points = 0;
  for (int i = 0; i < 512*512*1024; i++)
    if (scene_tsdf1[i] < tsdf_threshold)
      num_points++;

  std::string ptsfile = "pointcloud1_"+std::to_string(fuse_frame_start_idx_i) + "_" + std::to_string(fuse_frame_start_idx_i + num_frames_per_frag)+".ply";
  FILE *fp = fopen(ptsfile.c_str(), "w");
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


  ptsfile = "pointcloud2_"+std::to_string(fuse_frame_start_idx_j) + "_" + std::to_string(fuse_frame_start_idx_j + num_frames_per_frag)+".ply";
  fp = fopen(ptsfile.c_str(), "w");
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




  return 0;
}

