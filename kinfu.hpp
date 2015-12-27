#include <pwd.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <curl/curl.h>
#include <dirent.h>
#include <cerrno>
#include <algorithm>
#include <libio.h>
#include <ctime>

#include <png++/png.hpp>
#include <jpeglib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iomanip>
#include <sstream>

#include "ddd_demo/detect_keypoints.h"
#include "ddd_demo/ddd.h"
// #include "sensorData/sensorData.h"

// #include "cuda.h"
// #include "cuda_runtime.h"

// using namespace std;

////////////////////////////////////////////////////////////////////////////////

// void sys_command(std::string str) {
//   if (system(str.c_str()))
//     return;
// }

// ////////////////////////////////////////////////////////////////////////////////

size_t write_string(char *buf, size_t size, size_t nmemb, std::string *name) {
  (*name) += std::string(buf, size * nmemb);
  return size * nmemb;
}

// ////////////////////////////////////////////////////////////////////////////////

void get_server_filename(CURL *curl, std::string *url_name, std::string *name) {
  curl_easy_setopt(curl, CURLOPT_URL,           url_name->c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_string);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,     name);

  curl_easy_perform(curl);
}

// ////////////////////////////////////////////////////////////////////////////////

void parse_names(const std::string &names,
                 std::string ext,
                 const std::string &sun3d_dir,
                 const std::string &local_dir,
                 std::vector<std::string> *file_i,
                 std::vector<std::string> *file_o) {
  ext += "\"";

  unsigned int pos = names.find(ext);
  while ( pos < names.size()) {
    unsigned p = names.rfind("\"", pos);
    p++;

    std::string n = names.substr(p, pos - p + 4);
    file_i->push_back(sun3d_dir + n);
    file_o->push_back(local_dir + n);

    pos = names.find(ext, pos + ext.size());
  }
}

// ////////////////////////////////////////////////////////////////////////////////

size_t write_file(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t written = fwrite(ptr, size, nmemb, stream);
  return written;
}

// ////////////////////////////////////////////////////////////////////////////////

void write_filenames(CURL *curl, std::vector<std::string> *file_i, std::vector<std::string> *file_o) {
  for (size_t i = 0; i < file_i->size(); ++i) {
    std::cout << "    " << (*file_i)[i];
    FILE *fp = fopen((*file_o)[i].c_str(), "wb");

    curl_easy_setopt(curl, CURLOPT_URL,           (*file_i)[i].c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     fp);
    curl_easy_perform(curl);

    fclose(fp);
    std::cout << std::string((*file_i)[i].length() + 4, '\b');
  }
}

////////////////////////////////////////////////////////////////////////////////

void download_data( CURL *curl, std::string server_dir, const std::string &ext, const std::string &local_dir) {
  std::vector<std::string> file_i;
  std::vector<std::string> file_o;

  std::string names;

  get_server_filename(curl, &server_dir, &names);
  parse_names(names, ext, server_dir, local_dir, &file_i, &file_o);
  names.clear();
  std::cout << "Copying " << file_i.size() << " " << ext.substr(1) << " file(s) from server..." << std::endl;
  write_filenames(curl, &file_i, &file_o);
  std::cout << std::endl << "done!" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

void data_server_to_local(const std::string &sequence_name, const std::string &local_dir) {
  std::string sun3d_path   = "http://sun3d.cs.princeton.edu/data/" + sequence_name;

  std::string sun3d_camera = sun3d_path + "intrinsics.txt";
  std::string sun3d_image  = sun3d_path + "image/";
  std::string sun3d_depth  = sun3d_path + "depth/";
  std::string sun3d_pose   = sun3d_path + "extrinsics/";

  std::string local_camera = local_dir  + "intrinsics.txt";
  std::string local_image  = local_dir  + "image/";
  std::string local_depth  = local_dir  + "depth/";
  std::string local_pose   = local_dir  + "extrinsics/";

  std::string image_ext    = ".jpg";
  std::string depth_ext    = ".png";
  std::string pose_ext     = ".txt";

  sys_command( "mkdir -p " + local_dir);
  sys_command( "mkdir -p " + local_image);
  sys_command( "mkdir -p " + local_depth);
  sys_command( "mkdir -p " + local_pose);

  CURL* curl;
  curl_global_init(CURL_GLOBAL_ALL);
  curl = curl_easy_init();

  download_data(curl, sun3d_path,  pose_ext,  local_dir);
  download_data(curl, sun3d_image, image_ext, local_image);
  download_data(curl, sun3d_depth, depth_ext, local_depth);
  download_data(curl, sun3d_pose,  pose_ext,  local_pose);

  curl_easy_cleanup(curl);
  curl_global_cleanup();
}

////////////////////////////////////////////////////////////////////////////////

void get_local_filenames(const std::string &dir, std::vector<std::string> *file_list) {
  DIR *dp;
  struct dirent *dirp;
  if ((dp  = opendir(dir.c_str())) == NULL) {
    std::cout << "Error(" << errno << ") opening " << dir << std::endl;
  }

  while ((dirp = readdir(dp)) != NULL) {
    file_list->push_back(dir + std::string(dirp->d_name));
  }
  closedir(dp);

  sort( file_list->begin(), file_list->end() );
  file_list->erase(file_list->begin()); //.
  file_list->erase(file_list->begin()); //..
}

////////////////////////////////////////////////////////////////////////////////

const int kImageRows = 480;
const int kImageCols = 640;
const int kSampleFactor = 30;
const int kImageChannels = 3;
const int kFileNameLength = 24;
const int kTimeStampPos = 8;
const int kTimeStampLength = 12;

typedef unsigned char uchar;
typedef unsigned short ushort;

struct stat file_info;

struct _cam_k {
  float fx;
  float fy;
  float cx;
  float cy;
} cam_K;

//float cam_view_frustum[3][5];

typedef struct _extrinsic {
  float R[9];
  float t[3];
} extrinsic;

typedef struct _keypoint {
  float x;
  float y;
  float z;
  float response;
} keypoint;

typedef struct _normal {
  float x;
  float y;
  float z;
} normal;

struct _voxel_volume {
  float unit;
  float mu_grid;
  float mu;
  float size_grid[3];
  float range[3][2];
  float* tsdf;
  float* weight;
  std::vector<keypoint> keypoints;
} voxel_volume;

////////////////////////////////////////////////////////////////////////////////

bool get_depth_data_sun3d(const std::string &file_name, ushort *data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(), png::require_color_space< png::gray_pixel_16 >());

  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      ushort s = img.get_pixel(j, i);
      *(data + index) = (s << 13 | s >> 3);
      ++index;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool get_depth_data_sevenscenes(const std::string &file_name, ushort *data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(), png::require_color_space< png::gray_pixel_16 >());

  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      ushort s = img.get_pixel(j, i);
      *(data + index) = s;
      ++index;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool get_image_data_sun3d(const std::string &file_name, uchar *data) {
  unsigned char *raw_image = NULL;

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];

  FILE *infile = fopen(file_name.c_str(), "rb");
  unsigned long location = 0;

  if (!infile) {
    printf("Error opening jpeg file %s\n!", file_name.c_str());
    return -1;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  raw_image = (unsigned char*) malloc(
                cinfo.output_width * cinfo.output_height * cinfo.num_components);
  row_pointer[0] = (unsigned char *) malloc(
                     cinfo.output_width * cinfo.num_components);

  while (cinfo.output_scanline < cinfo.image_height) {
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
    for (uint i = 0; i < cinfo.image_width * cinfo.num_components; i++)
      raw_image[location++] = row_pointer[0][i];
  }

  int index = 0;
  for (uint i = 0; i < cinfo.image_height; ++i) {
    for (uint j = 0; j < cinfo.image_width; ++j) {
      for (int k = 0; k < kImageChannels; ++k) {
        *(data + index) = raw_image[(i * cinfo.image_width * 3) + (j * 3) + k];
        ++index;
      }
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(row_pointer[0]);
  fclose(infile);

  return true;
}

////////////////////////////////////////////////////////////////////////////////

void get_extrinsic_data(const std::string &file_name, int size,
                        std::vector<extrinsic> *poses) {
  FILE *fp = fopen(file_name.c_str(), "r");

  for (int i = 0; i < size; ++i) {
    extrinsic m;
    for (int d = 0; d < 3; ++d) {
      int iret;
      iret = fscanf(fp, "%f", &m.R[d * 3 + 0]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 1]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 2]);
      iret = fscanf(fp, "%f", &m.t[d]);
    }
    poses->push_back(m);
  }

  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

int get_timestamp(const std::string &file_name) {
  return atoi(file_name.substr(file_name.size() - kFileNameLength + kTimeStampPos, kTimeStampLength).c_str());
}

////////////////////////////////////////////////////////////////////////////////

void sync_depth(std::vector<std::string> image_list, std::vector<std::string> *depth_list) {
  std::vector<std::string> depth_temp;
  depth_temp.swap(*depth_list);
  depth_list->clear();
  depth_list->reserve(image_list.size());

  int idx = 0;
  int depth_time = get_timestamp(depth_temp[idx]);
  int time_low = depth_time;


  for (unsigned int i = 0; i < image_list.size(); ++i) {
    int image_time = get_timestamp(image_list[i]);

    while (depth_time < image_time) {
      if (idx == depth_temp.size() - 1)
        break;

      time_low = depth_time;
      depth_time = get_timestamp(depth_temp[++idx]);
    }

    if (idx == 0 && depth_time > image_time) {
      depth_list->push_back(depth_temp[idx]);
      continue;
    }

    if (std::abs(image_time - time_low) < std::abs(depth_time - image_time)) {
      depth_list->push_back(depth_temp[idx - 1]);
    } else {
      depth_list->push_back(depth_temp[idx]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void get_sun3d_data(std::string sequence_name, std::string local_dir, std::vector<std::string> &image_list, std::vector<std::string> &depth_list, std::vector<extrinsic> &extrinsic_poses) {

  // Check if local directory exists, if not, create it and populate it
  if (stat(local_dir.c_str(), &file_info) == 0 && (file_info.st_mode & S_IFDIR))
    std::cout << "Local directory for RGB-D sequence exists. Skipping file retrieval." << std::endl;
  else
    data_server_to_local(sequence_name, local_dir);

  // Retrieve list of images
  std::string local_image     = local_dir + "image/";
  get_local_filenames(local_image, &image_list);

  // Retrieve list of depth data
  std::string local_depth     = local_dir + "depth/";
  get_local_filenames(local_depth, &depth_list);

  // Sync images with depth data
  sync_depth(image_list, &depth_list);

  // Retrieve list of extrinsic files and extract poses
  std::string local_extrinsic = local_dir + "extrinsics/";
  std::vector<std::string> extrinsic_list;
  get_local_filenames(local_extrinsic, &extrinsic_list);
  get_extrinsic_data(extrinsic_list[extrinsic_list.size() - 1], image_list.size(), &extrinsic_poses);

}

////////////////////////////////////////////////////////////////////////////////

void get_sevenscenes_data(std::string local_dir, std::vector<std::string> &image_list, std::vector<std::string> &depth_list, std::vector<extrinsic> &extrinsic_poses) {

  // Get all filenames
  std::vector<std::string> local_filenames;
  get_local_filenames(local_dir, &local_filenames);

  // Sort filenames as iamge/depth/extrinsic
  std::vector<std::string> extrinsic_list;
  for (int i = 0; i < local_filenames.size(); i++) {
    if (local_filenames[i].find(".color.png") != std::string::npos)
      image_list.push_back(local_filenames[i]);
    if (local_filenames[i].find(".depth.png") != std::string::npos)
      depth_list.push_back(local_filenames[i]);
    if (local_filenames[i].find(".pose.txt") != std::string::npos)
      extrinsic_list.push_back(local_filenames[i]);
  }

  // Get extrinsic matrices
  for (int i = 0; i < extrinsic_list.size(); i++) {
    FILE *fp = fopen(extrinsic_list[i].c_str(), "r");
    int iret;
    extrinsic m;
    for (int d = 0; d < 3; ++d) {
      iret = fscanf(fp, "%f", &m.R[d * 3 + 0]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 1]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 2]);
      iret = fscanf(fp, "%f", &m.t[d]);
    }
    extrinsic_poses.push_back(m);
    fclose(fp);

  }
}

////////////////////////////////////////////////////////////////////////////////

void init_voxel_volume() {
  voxel_volume.unit = 0.01;
  voxel_volume.mu_grid = 5;
  voxel_volume.mu = voxel_volume.unit * voxel_volume.mu_grid;

  voxel_volume.size_grid[0] = 512;
  voxel_volume.size_grid[1] = 512;
  voxel_volume.size_grid[2] = 1024;

  voxel_volume.range[0][0] = -voxel_volume.size_grid[0] * voxel_volume.unit / 2;
  voxel_volume.range[0][1] = voxel_volume.range[0][0] + voxel_volume.size_grid[0] * voxel_volume.unit;
  voxel_volume.range[1][0] = -voxel_volume.size_grid[1] * voxel_volume.unit / 2;
  voxel_volume.range[1][1] = voxel_volume.range[1][0] + voxel_volume.size_grid[1] * voxel_volume.unit;
  voxel_volume.range[2][0] = -0.5;
  voxel_volume.range[2][1] = voxel_volume.range[2][0] + voxel_volume.size_grid[2] * voxel_volume.unit;

  // std::cout << voxel_volume.range[0][0] << std::endl;
  // std::cout << voxel_volume.range[1][0] << std::endl;
  // std::cout << voxel_volume.range[2][0] << std::endl;
  // std::cout << voxel_volume.range[0][1] << std::endl;
  // std::cout << voxel_volume.range[1][1] << std::endl;
  // std::cout << voxel_volume.range[2][1] << std::endl;

  voxel_volume.tsdf = new float[512 * 512 * 1024];
  voxel_volume.weight = new float[512 * 512 * 1024];
  memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
  for (int i = 0; i < 512 * 512 * 1024; i++)
    voxel_volume.tsdf[i] = 1.0f;

  voxel_volume.keypoints.clear();
}

////////////////////////////////////////////////////////////////////////////////

void save_volume_to_ply(const std::string &file_name) {
  float tsdf_threshold = 0.2;
  float weight_threshold = 1;
  float radius = 5;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold)
      num_points++;

  // std::cout << "keypoint size during check: " << voxel_volume.keypoints.size() << std::endl;
  int keypoint_count = 0;
  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points + (int)voxel_volume.keypoints.size() * 20);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < 512 * 512 * 1024; i++) {

    // if (voxel_volume.weight[i] > weight_threshold) {
    //   int z = floor(i/(512*512));
    //   int y = floor((i - (z*512*512))/512);
    //   int x = i - (z*512*512) - (y*512);
    //   float float_x = (float) x;
    //   float float_y = (float) y;
    //   float float_z = (float) z;
    //   fwrite(&float_x, sizeof(float), 1, fp);
    //   fwrite(&float_y, sizeof(float), 1, fp);
    //   fwrite(&float_z, sizeof(float), 1, fp);
    //   uchar r = (char)0;
    //   uchar g = (char)0;
    //   uchar b = (char)255;
    //   fwrite(&r, sizeof(uchar), 1, fp);
    //   fwrite(&g, sizeof(uchar), 1, fp);
    //   fwrite(&b, sizeof(uchar), 1, fp);
    // }

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (512 * 512));
      int y = floor((i - (z * 512 * 512)) / 512);
      int x = i - (z * 512 * 512) - (y * 512);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      // If voxel is keypoint, color it red, otherwise color it gray
      uchar r = (uchar)180;
      uchar g = (uchar)180;
      uchar b = (uchar)180;
      bool is_keypoint = false;
      // bool is_valid_keypoint = false;
      for (int k = 0; k < voxel_volume.keypoints.size(); k++) {
        if (voxel_volume.keypoints[k].x == x && voxel_volume.keypoints[k].y == y && voxel_volume.keypoints[k].z == z) {
          r = (uchar)0;
          g = (uchar)0;
          b = (uchar)255;
          // float empty_space = 0;
          float occupancy = 0;
          for (int kk = -15; kk <= 15; kk++) {
            for (int jj = -15; jj <= 15; jj++) {
              for (int ii = -15; ii <= 15; ii++) {
                if (voxel_volume.weight[(z + kk) * 512 * 512 + (y + jj) * 512 + (x + ii)] >= 1) {
                  occupancy++;
                }
              }
            }
          }
          //float occupancy = 1 - empty_space/(31*31*31);
          occupancy = occupancy / (31 * 31 * 31);
          if (occupancy > 0.5) {
            r = (uchar)255;
            g = (uchar)0;
            b = (uchar)0;
            keypoint_count++;
            // is_valid_keypoint = true;
          }
          is_keypoint = true;
          break;
        }
      }

      fwrite(&r, sizeof(uchar), 1, fp);
      fwrite(&g, sizeof(uchar), 1, fp);
      fwrite(&b, sizeof(uchar), 1, fp);

      // Draw 5x5x5 box around keypoint
      if (is_keypoint) {
        for (int kk = -1; kk <= 1; kk++) {
          for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
              int num_edges = 0;
              if (kk == -1 || kk == 1)
                num_edges++;
              if (jj == -1 || jj == 1)
                num_edges++;
              if (ii == -1 || ii == 1)
                num_edges++;
              if (num_edges >= 2) {
                float float_x = (float) x + ii;
                float float_y = (float) y + jj;
                float float_z = (float) z + kk;
                fwrite(&float_x, sizeof(float), 1, fp);
                fwrite(&float_y, sizeof(float), 1, fp);
                fwrite(&float_z, sizeof(float), 1, fp);
                fwrite(&r, sizeof(uchar), 1, fp);
                fwrite(&g, sizeof(uchar), 1, fp);
                fwrite(&b, sizeof(uchar), 1, fp);
              }
            }
          }
        }
      }

    }
  }
  // std::cout << keypoint_count << std::endl;
  fclose(fp);
}

void save_iclnuim_volume_to_ply(const std::string &file_name) {
  float tsdf_threshold = 0.2;
  float weight_threshold = 1;
  float radius = 5;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold)
      num_points++;

  // std::cout << "keypoint size during check: " << voxel_volume.keypoints.size() << std::endl;
  int keypoint_count = 0;
  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points + (int)voxel_volume.keypoints.size() * 20);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < 512 * 512 * 1024; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (512 * 512));
      int y = floor((i - (z * 512 * 512)) / 512);
      int x = i - (z * 512 * 512) - (y * 512);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      // If voxel is keypoint, color it red, otherwise color it gray
      uchar r = (uchar)180;
      uchar g = (uchar)180;
      uchar b = (uchar)180;
      bool is_keypoint = false;
      // bool is_valid_keypoint = false;
      for (int k = 0; k < voxel_volume.keypoints.size(); k++) {
        if (voxel_volume.keypoints[k].x == x && voxel_volume.keypoints[k].y == y && voxel_volume.keypoints[k].z == z) {
          r = (uchar)0;
          g = (uchar)0;
          b = (uchar)255;
          // float empty_space = 0;
          float occupancy = 0;
          for (int kk = -15; kk <= 15; kk++) {
            for (int jj = -15; jj <= 15; jj++) {
              for (int ii = -15; ii <= 15; ii++) {
                if (voxel_volume.weight[(z + kk) * 512 * 512 + (y + jj) * 512 + (x + ii)] >= 1) {
                  occupancy++;
                }
              }
            }
          }
          //float occupancy = 1 - empty_space/(31*31*31);
          occupancy = occupancy / (31 * 31 * 31);
          if (occupancy > 0.5) {
            r = (uchar)255;
            g = (uchar)0;
            b = (uchar)0;
            keypoint_count++;
            // is_valid_keypoint = true;
          }
          is_keypoint = true;
          break;
        }
      }

      fwrite(&r, sizeof(uchar), 1, fp);
      fwrite(&g, sizeof(uchar), 1, fp);
      fwrite(&b, sizeof(uchar), 1, fp);

      // Draw 5x5x5 box around keypoint
      if (is_keypoint) {
        for (int kk = -1; kk <= 1; kk++) {
          for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
              int num_edges = 0;
              if (kk == -1 || kk == 1)
                num_edges++;
              if (jj == -1 || jj == 1)
                num_edges++;
              if (ii == -1 || ii == 1)
                num_edges++;
              if (num_edges >= 2) {
                float float_x = (float) x + ii;
                float float_y = (float) y + jj;
                float float_z = (float) z + kk;
                fwrite(&float_x, sizeof(float), 1, fp);
                fwrite(&float_y, sizeof(float), 1, fp);
                fwrite(&float_z, sizeof(float), 1, fp);
                fwrite(&r, sizeof(uchar), 1, fp);
                fwrite(&g, sizeof(uchar), 1, fp);
                fwrite(&b, sizeof(uchar), 1, fp);
              }
            }
          }
        }
      }

    }
  }
  // std::cout << keypoint_count << std::endl;
  fclose(fp);
}

void save_volume_to_tsdf(const std::string &file_name) {
  float tsdf_threshold = 0.2;
  float weight_threshold = 1;

  std::ofstream outFile(file_name, std::ios::binary | std::ios::out);

  // Find tight bounds of volume (bind to where weight > 1 and std::abs tsdf < 1)
  int start_idx = -1;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold && start_idx == -1) {
      start_idx = i;
      break;
    }

  int end_idx = -1;
  for (int i = 512 * 512 * 1024 - 1; i >= 0; i--)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold && end_idx == -1) {
      end_idx = i;
      break;
    }

  // std::cout << start_idx << std::endl;
  // std::cout << end_idx << std::endl;

  int num_elements = end_idx - start_idx + 1;
  outFile.write((char*)&start_idx, sizeof(int));
  outFile.write((char*)&num_elements, sizeof(int));

  for (int i = start_idx; i <= end_idx; i++)
    outFile.write((char*)&voxel_volume.tsdf[i], sizeof(float));

  outFile.close();
  // if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold)
  //   num_points++;

}

////////////////////////////////////////////////////////////////////////////////

void integrate(ushort* depth_data, float* range_grid, float* cam_R, float* cam_t) {

  for (int z = range_grid[2 * 2 + 0]; z < range_grid[2 * 2 + 1]; z++) {
    for (int y = range_grid[1 * 2 + 0]; y < range_grid[1 * 2 + 1]; y++) {
      for (int x = range_grid[0 * 2 + 0]; x < range_grid[0 * 2 + 1]; x++) {

        // grid to world coords
        float tmp_pos[3] = {0};
        tmp_pos[0] = (x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
        tmp_pos[1] = (y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
        tmp_pos[2] = (z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

        // transform
        float tmp_arr[3] = {0};
        tmp_arr[0] = tmp_pos[0] - cam_t[0];
        tmp_arr[1] = tmp_pos[1] - cam_t[1];
        tmp_arr[2] = tmp_pos[2] - cam_t[2];
        tmp_pos[0] = cam_R[0 * 3 + 0] * tmp_arr[0] + cam_R[1 * 3 + 0] * tmp_arr[1] + cam_R[2 * 3 + 0] * tmp_arr[2];
        tmp_pos[1] = cam_R[0 * 3 + 1] * tmp_arr[0] + cam_R[1 * 3 + 1] * tmp_arr[1] + cam_R[2 * 3 + 1] * tmp_arr[2];
        tmp_pos[2] = cam_R[0 * 3 + 2] * tmp_arr[0] + cam_R[1 * 3 + 2] * tmp_arr[1] + cam_R[2 * 3 + 2] * tmp_arr[2];
        if (tmp_pos[2] <= 0)
          continue;

        int px = std::round(cam_K.fx * (tmp_pos[0] / tmp_pos[2]) + cam_K.cx);
        int py = std::round(cam_K.fy * (tmp_pos[1] / tmp_pos[2]) + cam_K.cy);
        if (px < 1 || px > 640 || py < 1 || py > 480)
          continue;

        float p_depth = *(depth_data + (py - 1) * kImageCols + (px - 1)) / 1000.f;
        if (p_depth > 6.5f)
          continue;
        if (std::round(p_depth * 1000.0f) == 0)
          continue;

        float eta = (p_depth - tmp_pos[2]) * sqrt(1 + pow((tmp_pos[0] / tmp_pos[2]), 2) + pow((tmp_pos[1] / tmp_pos[2]), 2));
        if (eta <= -voxel_volume.mu)
          continue;

        // Integrate
        int volumeIDX = z * 512 * 512 + y * 512 + x;
        float sdf = std::min(1.0f, eta / voxel_volume.mu);
        float w_old = voxel_volume.weight[volumeIDX];
        float w_new = w_old + 1.0f;
        voxel_volume.weight[volumeIDX] = w_new;
        voxel_volume.tsdf[volumeIDX] = (voxel_volume.tsdf[volumeIDX] * w_old + sdf) / w_new;

      }
    }
  }

}

////////////////////////////////////////////////////////////////////////////////

// Save intrinsic matrix to global variable cam_K
void get_intrinsic_matrix(const std::string &local_camera) {
  int iret;
  float ff;
  FILE *fp = fopen(local_camera.c_str(), "r");
  iret = fscanf(fp, "%f", &cam_K.fx);
  iret = fscanf(fp, "%f", &ff);
  iret = fscanf(fp, "%f", &cam_K.cx);
  iret = fscanf(fp, "%f", &ff);
  iret = fscanf(fp, "%f", &cam_K.fy);
  iret = fscanf(fp, "%f", &cam_K.cy);
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void mulMatrix(const float m1[16], const float m2[16], float mOut[16]) {
  mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
  mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
  mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
  mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

  mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
  mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
  mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
  mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

  mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
  mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

////////////////////////////////////////////////////////////////////////////////

bool invMatrix(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;
  inv[0] = m[5]  * m[10] * m[15] -
           m[5]  * m[11] * m[14] -
           m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] -
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] +
           m[4]  * m[11] * m[14] +
           m[8]  * m[6]  * m[15] -
           m[8]  * m[7]  * m[14] -
           m[12] * m[6]  * m[11] +
           m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] -
           m[4]  * m[11] * m[13] -
           m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] +
            m[4]  * m[10] * m[13] +
            m[8]  * m[5] * m[14] -
            m[8]  * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] +
           m[1]  * m[11] * m[14] +
           m[9]  * m[2] * m[15] -
           m[9]  * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] -
           m[0]  * m[11] * m[14] -
           m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] +
           m[0]  * m[11] * m[13] +
           m[8]  * m[1] * m[15] -
           m[8]  * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] -
            m[0]  * m[10] * m[13] -
            m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] -
           m[1]  * m[7] * m[14] -
           m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] +
           m[0]  * m[7] * m[14] +
           m[4]  * m[2] * m[15] -
           m[4]  * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] -
            m[0]  * m[7] * m[13] -
            m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] +
            m[0]  * m[6] * m[13] +
            m[4]  * m[1] * m[14] -
            m[4]  * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

////////////////////////////////////////////////////////////////////////////////

void compute_frustum_bounds(std::vector<extrinsic> &extrinsic_poses, int base_frame, int curr_frame, float* cam_R, float* cam_t, float* range_grid) {

  // if (curr_frame == 0) {

  //   // Relative rotation of first frame to first frame is identity matrix
  //   cam_R[0] = 1.0f;
  //   cam_R[1] = 0.0f;
  //   cam_R[2] = 0.0f;
  //   cam_R[3] = 0.0f;
  //   cam_R[4] = 1.0f;
  //   cam_R[5] = 0.0f;
  //   cam_R[6] = 0.0f;
  //   cam_R[7] = 0.0f;
  //   cam_R[8] = 1.0f;

  // } else {

  // Use two extrinsic matrices to compute relative rotations between current frame and first frame
  extrinsic ex_pose1 = extrinsic_poses[base_frame];
  extrinsic ex_pose2 = extrinsic_poses[curr_frame];

  float ex_mat1[16] =
  { ex_pose1.R[0 * 3 + 0], ex_pose1.R[0 * 3 + 1], ex_pose1.R[0 * 3 + 2], ex_pose1.t[0],
    ex_pose1.R[1 * 3 + 0], ex_pose1.R[1 * 3 + 1], ex_pose1.R[1 * 3 + 2], ex_pose1.t[1],
    ex_pose1.R[2 * 3 + 0], ex_pose1.R[2 * 3 + 1], ex_pose1.R[2 * 3 + 2], ex_pose1.t[2],
    0,                0,                0,            1
  };

  float ex_mat2[16] =
  { ex_pose2.R[0 * 3 + 0], ex_pose2.R[0 * 3 + 1], ex_pose2.R[0 * 3 + 2], ex_pose2.t[0],
    ex_pose2.R[1 * 3 + 0], ex_pose2.R[1 * 3 + 1], ex_pose2.R[1 * 3 + 2], ex_pose2.t[1],
    ex_pose2.R[2 * 3 + 0], ex_pose2.R[2 * 3 + 1], ex_pose2.R[2 * 3 + 2], ex_pose2.t[2],
    0,                 0,                0,            1
  };

  float ex_mat1_inv[16] = {0};
  invMatrix(ex_mat1, ex_mat1_inv);
  float ex_mat_rel[16] = {0};
  mulMatrix(ex_mat1_inv, ex_mat2, ex_mat_rel);

  cam_R[0] = ex_mat_rel[0];
  cam_R[1] = ex_mat_rel[1];
  cam_R[2] = ex_mat_rel[2];
  cam_R[3] = ex_mat_rel[4];
  cam_R[4] = ex_mat_rel[5];
  cam_R[5] = ex_mat_rel[6];
  cam_R[6] = ex_mat_rel[8];
  cam_R[7] = ex_mat_rel[9];
  cam_R[8] = ex_mat_rel[10];

  cam_t[0] = ex_mat_rel[3];
  cam_t[1] = ex_mat_rel[7];
  cam_t[2] = ex_mat_rel[11];

  // }

  // std::cout << cam_R[0] << std::endl;
  // std::cout << cam_R[1] << std::endl;
  // std::cout << cam_R[2] << std::endl;
  // std::cout << cam_R[3] << std::endl;
  // std::cout << cam_R[4] << std::endl;
  // std::cout << cam_R[5] << std::endl;
  // std::cout << cam_R[6] << std::endl;
  // std::cout << cam_R[7] << std::endl;
  // std::cout << cam_R[8] << std::endl;

  // std::cout << cam_t[0] << std::endl;
  // std::cout << cam_t[1] << std::endl;
  // std::cout << cam_t[2] << std::endl;

  // Init cam view frustum
  float cam_view_frustum[15] =
  { 0, -320 * 8 / cam_K.fx, -320 * 8 / cam_K.fx, 320 * 8 / cam_K.fx,  320 * 8 / cam_K.fx,
    0, -240 * 8 / cam_K.fx,  240 * 8 / cam_K.fx, 240 * 8 / cam_K.fx, -240 * 8 / cam_K.fx,
    0,               8,               8,              8,              8
  };

  // Rotate cam view frustum wrt Rt
  for (int i = 0; i < 5; i++) {
    float tmp_arr[3] = {0};
    tmp_arr[0] = cam_R[0 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[0 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[0 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[1] = cam_R[1 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[1 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[1 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[2] = cam_R[2 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[2 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[2 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    cam_view_frustum[0 * 5 + i] = tmp_arr[0] + cam_t[0];
    cam_view_frustum[1 * 5 + i] = tmp_arr[1] + cam_t[1];
    cam_view_frustum[2 * 5 + i] = tmp_arr[2] + cam_t[2];
  }

  // std::cout << cam_view_frustum[0*5+0] << std::endl;
  // std::cout << cam_view_frustum[0*5+1] << std::endl;
  // std::cout << cam_view_frustum[0*5+2] << std::endl;
  // std::cout << cam_view_frustum[0*5+3] << std::endl;
  // std::cout << cam_view_frustum[0*5+4] << std::endl;

  // std::cout << cam_view_frustum[1*5+0] << std::endl;
  // std::cout << cam_view_frustum[1*5+1] << std::endl;
  // std::cout << cam_view_frustum[1*5+2] << std::endl;
  // std::cout << cam_view_frustum[1*5+3] << std::endl;
  // std::cout << cam_view_frustum[1*5+4] << std::endl;

  // std::cout << cam_view_frustum[2*5+0] << std::endl;
  // std::cout << cam_view_frustum[2*5+1] << std::endl;
  // std::cout << cam_view_frustum[2*5+2] << std::endl;
  // std::cout << cam_view_frustum[2*5+3] << std::endl;
  // std::cout << cam_view_frustum[2*5+4] << std::endl;

  // Compute frustum endpoints
  float range2test[3][2] = {0};
  for (int i = 0; i < 3; i++) {
    range2test[i][0] = *std::min_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
    range2test[i][1] = *std::max_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
  }

  // std::cout << range2test[0][0] << std::endl;
  // std::cout << range2test[1][0] << std::endl;
  // std::cout << range2test[2][0] << std::endl;

  // std::cout << range2test[0][1] << std::endl;
  // std::cout << range2test[1][1] << std::endl;
  // std::cout << range2test[2][1] << std::endl;

  // Compute frustum bounds wrt volume
  for (int i = 0; i < 3; i++) {
    range_grid[i * 2 + 0] = std::max(0.0f, std::floor((range2test[i][0] - voxel_volume.range[i][0]) / voxel_volume.unit));
    range_grid[i * 2 + 1] = std::min(voxel_volume.size_grid[i], std::ceil((range2test[i][1] - voxel_volume.range[i][0]) / voxel_volume.unit + 1));
  }

  // std::cout << range_grid[0*2+0] << std::endl;
  // std::cout << range_grid[0*2+1] << std::endl;
  // std::cout << range_grid[1*2+0] << std::endl;
  // std::cout << range_grid[1*2+1] << std::endl;
  // std::cout << range_grid[2*2+0] << std::endl;
  // std::cout << range_grid[2*2+1] << std::endl;

}

////////////////////////////////////////////////////////////////////////////////

normal comput_tsdf_normal(int x, int y, int z, float weight) {
  float weight_threshold = 1;

  float SDFx1 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z) * 512 * 512 + (y) * 512 + (x + 1)] > weight_threshold)
    SDFx1 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x + 1)];
  float SDFx2 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z) * 512 * 512 + (y) * 512 + (x - 1)] > weight_threshold)
    SDFx2 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x - 1)];

  float SDFy1 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z) * 512 * 512 + (y + 1) * 512 + (x)] > weight_threshold)
    SDFy1 = voxel_volume.tsdf[(z) * 512 * 512 + (y + 1) * 512 + (x)];
  float SDFy2 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z) * 512 * 512 + (y - 1) * 512 + (x)] > weight_threshold)
    SDFy2 = voxel_volume.tsdf[(z) * 512 * 512 + (y - 1) * 512 + (x)];

  float SDFz1 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z + 1) * 512 * 512 + (y) * 512 + (x)] > weight_threshold)
    SDFz1 = voxel_volume.tsdf[(z + 1) * 512 * 512 + (y) * 512 + (x)];
  float SDFz2 = voxel_volume.tsdf[(z) * 512 * 512 + (y) * 512 + (x)];
  if (voxel_volume.weight[(z - 1) * 512 * 512 + (y) * 512 + (x)] > weight_threshold)
    SDFz2 = voxel_volume.tsdf[(z - 1) * 512 * 512 + (y) * 512 + (x)];

  normal tmp_norm;
  tmp_norm.x = SDFx1 - SDFx2;
  tmp_norm.y = SDFy1 - SDFy2;
  tmp_norm.z = SDFz1 - SDFz2;

  if (tmp_norm.x == 0 && tmp_norm.y == 0 && tmp_norm.z == 0)
    return tmp_norm;

  float denum = sqrt(tmp_norm.x * tmp_norm.x + tmp_norm.y * tmp_norm.y + tmp_norm.z * tmp_norm.z);
  tmp_norm.x = weight * tmp_norm.x / denum;
  tmp_norm.y = weight * tmp_norm.y / denum;
  tmp_norm.z = weight * tmp_norm.z / denum;
  return tmp_norm;

}

////////////////////////////////////////////////////////////////////////////////

bool compute_normal_covariance(int x, int y, int z, float tsdf_threshold, float weight_threshold, float* covariance) {

  int radius = 5;
  // int volume_idx = z * 512 * 512 + y * 512 + x;
  std::vector<normal> normals;

  for (int k = -radius; k <= radius; k++) {
    for (int j = -radius; j <= radius; j++) {
      for (int i = -radius; i <= radius; i++) {
        // if (voxel_volume.weight[(z+k)*512*512 + (y+j)*512 + (x+i)] < weight_threshold)
        //   return false;
        if (std::abs(voxel_volume.tsdf[(z + k) * 512 * 512 + (y + j) * 512 + (x + i)]) < tsdf_threshold && voxel_volume.weight[(z + k) * 512 * 512 + (y + j) * 512 + (x + i)] > weight_threshold) {
          float ksize = (float)radius * 2.0f + 1.0f;
          float sigma = 0.3f * (ksize / 2.0f - 1.0f) + 0.8f;
          float incDist = std::max(std::abs((float)i), std::max(std::abs((float)j), std::abs((float)k)));
          float gaussianWeight = exp(-(incDist * incDist) / (2.0f * sigma * sigma)) / (sqrt(2.0f * 3.1415927) * sigma);
          normal tmp_norm = comput_tsdf_normal(x + i, y + j, z + k, gaussianWeight * 255.0f);
          normals.push_back(tmp_norm);
          // normals.push_back(gaussianWeight*255*computeSurfaceNormal(x + i, y + j, z + k));
        }
      }
    }
  }

  if (normals.size() < (radius * 2 + 1) * (radius * 2 + 1))
    return false;

  // std::cout << "size of normals: " << normals.size() << std::endl;

  float Ixx = 0; float Ixy = 0; float Ixz = 0;
  float Iyy = 0; float Iyz = 0; float Izz = 0;
  for (int i = 0; i < normals.size(); i++) {
    Ixx = Ixx + normals[i].x * normals[i].x;
    Ixy = Ixy + normals[i].x * normals[i].y;
    Ixz = Ixz + normals[i].x * normals[i].z;
    Iyy = Iyy + normals[i].y * normals[i].y;
    Iyz = Iyz + normals[i].y * normals[i].z;
    Izz = Izz + normals[i].z * normals[i].z;
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

////////////////////////////////////////////////////////////////////////////////

void find_keypoints(float *range_grid) {

  float tsdf_threshold = 0.2;
  float weight_threshold = 1;
  float response_threshold = 1.0f;
  int radius = 5;

  float* responses = new float[512 * 512 * 1024];
  memset(responses, 0, sizeof(float) * 512 * 512 * 1024);


  // for (int z = range_grid[2*2+0]; z < range_grid[2*2+1]; z++) {
  //   for (int y = range_grid[1*2+0]; y < range_grid[1*2+1]; y++) {
  //     for (int x = range_grid[0*2+0]; x < range_grid[0*2+1]; x++) {

  for (int z = range_grid[2 * 2 + 0] + radius + 1; z < range_grid[2 * 2 + 1] - radius - 1; z++) {
    for (int y = range_grid[1 * 2 + 0] + radius + 1; y < range_grid[1 * 2 + 1] - radius - 1; y++) {
      for (int x = range_grid[0 * 2 + 0] + radius + 1; x < range_grid[0 * 2 + 1] - radius - 1; x++) {
        int volume_idx = z * 512 * 512 + y * 512 + x;
        if (std::abs(voxel_volume.tsdf[volume_idx]) < tsdf_threshold && voxel_volume.weight[volume_idx] > weight_threshold) {
          float covar[9];
          bool isValid = compute_normal_covariance(x, y, z, tsdf_threshold, weight_threshold, covar);

          float response = 0;

          if (isValid) {
            // for (int i = 0; i < 9; i++)
            //   std::cout << covariance[i] << std::endl;

            // Compute harris corner response
            float covar_det = covar[0] * covar[4] * covar[8] + covar[1] * covar[5] * covar[6] + covar[2] * covar[3] * covar[7] -
                              covar[6] * covar[4] * covar[2] - covar[7] * covar[5] * covar[0] - covar[8] * covar[3] * covar[1];
            float covar_trace = covar[0] + covar[4] + covar[8];
            if (covar_trace != 0)
              response = 0.04f + covar_det - 0.04f * covar_trace * covar_trace;

            // std::cout << response << std::endl;
          }

          responses[volume_idx] = response;

          // std::cout << std::endl;

        }
      }
    }
  }

  // Loop through each point and check if it is a local maximum
  for (int z = radius + 1; z < 1024 - radius - 1; z++) {
    for (int y = radius + 1; y < 512 - radius - 1; y++) {
      for (int x = radius + 1; x < 512 - radius - 1; x++) {
        int volume_idx = z * 512 * 512 + y * 512 + x;
        if (responses[volume_idx] < response_threshold)
          continue;
        bool isCorner = true;
        for (int k = -radius; k <= radius; k++)
          for (int j = -radius; j <= radius; j++)
            for (int i = -radius; i <= radius; i++) {
              if (isCorner && responses[(z + k) * 512 * 512 + (y + j) * 512 + (x + i)] > responses[volume_idx])
                isCorner = false;
            }

        // If point is a local maximum, add as keypoint
        if (isCorner) {
          keypoint tmp_keypt;
          tmp_keypt.x = x;
          tmp_keypt.y = y;
          tmp_keypt.z = z;
          tmp_keypt.response = responses[volume_idx];
          voxel_volume.keypoints.push_back(tmp_keypt);
        }
      }
    }
  }

  // std::cout << "number of keypoints: " << voxel_volume.keypoints.size() << std::endl;

  delete [] responses;

}

void save_volume_keypoints(const std::string &file_name) {

  std::vector<keypoint> valid_keypoints;

  // Find valid points
  for (int i = 0; i < voxel_volume.keypoints.size(); i++) {
    float occupancy = 0;
    // float empty_space = 0;
    // std::cout << voxel_volume.keypoints[i].x << " " << voxel_volume.keypoints[i].y << " " << voxel_volume.keypoints[i].z << std::endl;
    for (int k = -15; k <= 15; k++) {
      for (int j = -15; j <= 15; j++) {
        for (int ii = -15; ii <= 15; ii++) {
          int volume_idx = (((int)(voxel_volume.keypoints[i].z)) + k) * 512 * 512 + (((int)(voxel_volume.keypoints[i].y) + j)) * 512 + (((int)(voxel_volume.keypoints[i].x)) + ii);
          if (volume_idx >= 0 && volume_idx < 512 * 512 * 1024) {
            //std::cout << voxel_volume.weight[volume_idx]  << std::endl;
            if (voxel_volume.weight[volume_idx] >= 1) {
              occupancy++;
            }
          }
        }
      }
    }
    occupancy = occupancy / (31 * 31 * 31);
    // std::cout << occupancy << std::endl;
    if (occupancy > 0.5) {
      keypoint tmp_keypt = voxel_volume.keypoints[i];
      valid_keypoints.push_back(tmp_keypt);
    }
  }

  // std::cout << valid_keypoints.size() << " " << voxel_volume.keypoints.size() << std::endl;

  FILE *fp = fopen(file_name.c_str(), "w");

  int num_points = valid_keypoints.size();
  fprintf(fp, "%d\n", num_points);

  for (int i = 0; i < num_points; i++) {
    fprintf(fp, "%f %f %f\n", valid_keypoints[i].x, valid_keypoints[i].y, valid_keypoints[i].z);
  }

  fclose(fp);


}

void save_volume_to_world_matrix(const std::string &file_name, std::vector<extrinsic> &extrinsic_poses, int first_frame) {

  extrinsic ex_pose1 = extrinsic_poses[0];
  extrinsic ex_pose2 = extrinsic_poses[first_frame];

  float ex_mat1[16] =
  { ex_pose1.R[0 * 3 + 0], ex_pose1.R[0 * 3 + 1], ex_pose1.R[0 * 3 + 2], ex_pose1.t[0],
    ex_pose1.R[1 * 3 + 0], ex_pose1.R[1 * 3 + 1], ex_pose1.R[1 * 3 + 2], ex_pose1.t[1],
    ex_pose1.R[2 * 3 + 0], ex_pose1.R[2 * 3 + 1], ex_pose1.R[2 * 3 + 2], ex_pose1.t[2],
    0,                0,                0,            1
  };

  float ex_mat2[16] =
  { ex_pose2.R[0 * 3 + 0], ex_pose2.R[0 * 3 + 1], ex_pose2.R[0 * 3 + 2], ex_pose2.t[0],
    ex_pose2.R[1 * 3 + 0], ex_pose2.R[1 * 3 + 1], ex_pose2.R[1 * 3 + 2], ex_pose2.t[1],
    ex_pose2.R[2 * 3 + 0], ex_pose2.R[2 * 3 + 1], ex_pose2.R[2 * 3 + 2], ex_pose2.t[2],
    0,                 0,                0,            1
  };

  float ex_mat1_inv[16] = {0};
  invMatrix(ex_mat1, ex_mat1_inv);
  float ex_mat_rel[16] = {0};
  mulMatrix(ex_mat1_inv, ex_mat2, ex_mat_rel);

  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "%f %f %f %f\n", ex_mat_rel[0], ex_mat_rel[1], ex_mat_rel[2], ex_mat_rel[3]);
  fprintf(fp, "%f %f %f %f\n", ex_mat_rel[4], ex_mat_rel[5], ex_mat_rel[6], ex_mat_rel[7]);
  fprintf(fp, "%f %f %f %f\n", ex_mat_rel[8], ex_mat_rel[9], ex_mat_rel[10], ex_mat_rel[11]);
  fprintf(fp, "%f %f %f %f\n", ex_mat_rel[12], ex_mat_rel[13], ex_mat_rel[14], ex_mat_rel[15]);
  fclose(fp);

  // extrinsic ex_pose = extrinsic_poses[first_frame];

  // float ex_mat[16] =
  //   {ex_pose.R[0*3+0], ex_pose.R[0*3+1], ex_pose.R[0*3+2], ex_pose.t[0],
  //    ex_pose.R[1*3+0], ex_pose.R[1*3+1], ex_pose.R[1*3+2], ex_pose.t[1],
  //    ex_pose.R[2*3+0], ex_pose.R[2*3+1], ex_pose.R[2*3+2], ex_pose.t[2],
  //                   0,                0,                0,            1};

  // FILE *fp = fopen(file_name.c_str(), "w");
  // fprintf(fp, "%f %f %f %f\n", ex_mat[0], ex_mat[1], ex_mat[2], ex_mat[3]);
  // fprintf(fp, "%f %f %f %f\n", ex_mat[4], ex_mat[5], ex_mat[6], ex_mat[7]);
  // fprintf(fp, "%f %f %f %f\n", ex_mat[8], ex_mat[9], ex_mat[10], ex_mat[11]);
  // fprintf(fp, "%f %f %f %f\n", ex_mat[12], ex_mat[13], ex_mat[14], ex_mat[15]);
  // fclose(fp);

}

void generate_data_sun3d(const std::string &sequence_prefix, const std::string &local_data_dir, int start_idx, int end_idx, int num_frames_per_frag) {

  std::vector<std::string> image_list;
  std::vector<std::string> depth_list;
  std::vector<extrinsic> extrinsic_poses;

/////////////////////////////////////////////////////////////////////

  // Retrieve local directory
  std::string sequence_name = "mit_32_d507/d507_2/";
  passwd* pw = getpwuid(getuid());
  std::string home_dir(pw->pw_dir);
  std::string local_dir = local_data_dir + sequence_name;

  // Query RGB-D sequence from SUN3D
  get_sun3d_data(sequence_name, local_dir, image_list, depth_list, extrinsic_poses);

/////////////////////////////////////////////////////////////////////

  // Count total number of frames
  int total_files = 0;
  image_list.size() < depth_list.size() ? total_files = image_list.size() : total_files = depth_list.size();

  // Init voxel volume params
  init_voxel_volume();

  // Init intrinsic matrix K
  std::string local_camera = local_dir + "intrinsics.txt";
  if (std::ifstream(local_camera))
    get_intrinsic_matrix(local_camera);
  else {
    std::cout << "Intrinsics not found. Loading default matrix." << std::endl;
    cam_K.fx = 585.0f; cam_K.fy = 585.0f; cam_K.cx = 320.0f; cam_K.cy = 240.0f;
  }

  // Set first frame of sequence as base coordinate frame
  int first_frame = start_idx;

  // Fuse frames
  for (int curr_frame = start_idx; curr_frame <= end_idx; curr_frame++) {
    std::cout << "Fusing frame " << curr_frame << "...";

    // Load image/depth/extrinsic data for frame curr_frame
    // uchar *image_data = (uchar *) malloc(kImageRows * kImageCols * kImageChannels * sizeof(uchar));
    ushort *depth_data = (ushort *) malloc(kImageRows * kImageCols * sizeof(ushort));
    // get_image_data_sun3d(image_list[curr_frame], image_data);
    get_depth_data_sun3d(depth_list[curr_frame], depth_data);
    // std::cout << depth_list[curr_frame] << std::endl;
    // get_depth_data_sevenscenes(depth_list[curr_frame], depth_data);

    // if (curr_frame == 0) {
    //   int bestValue = 0;
    //   for (int i = 0; i < kImageRows * kImageCols ; i++)
    //     if ((int)depth_data[i] > bestValue)
    //       bestValue = (int)depth_data[i];
    //   std::cout << bestValue << std::endl;
    // }

    // Compute camera Rt between current frame and base frame (saved in cam_R and cam_t)
    // then return view frustum bounds for volume (saved in range_grid)
    float cam_R[9] = {0};
    float cam_t[3] = {0};
    float range_grid[6] = {0};
    compute_frustum_bounds(extrinsic_poses, first_frame, curr_frame, cam_R, cam_t, range_grid);

    // Integrate
    time_t tstart, tend;
    tstart = time(0);
    integrate(depth_data, range_grid, cam_R, cam_t);
    tend = time(0);

    // Clear memory
    // free(image_data);
    free(depth_data);

    // Print time
    std::cout << " done!" << std::endl;
    // std::cout << "Integrating took " << difftime(tend, tstart) << " second(s)." << std::endl;

    // Compute intersection between view frustum and current volume
    float view_occupancy = (range_grid[1] - range_grid[0]) * (range_grid[3] - range_grid[2]) * (range_grid[5] - range_grid[4]) / (512 * 512 * 1024);
    // std::cout << "Intersection: " << 100 * view_occupancy << "%" << std::endl;

    if ((curr_frame - first_frame >= num_frames_per_frag) || curr_frame == total_files - 1) {

      // Find keypoints
      find_keypoints(range_grid);

      // Save curr volume to file
      std::string volume_name = sequence_prefix + "scene" +  std::to_string(first_frame) + "_" + std::to_string(curr_frame);

      std::string scene_ply_name = volume_name + ".ply";
      save_volume_to_ply(scene_ply_name);

      std::string scene_tsdf_name = volume_name + ".tsdf";
      save_volume_to_tsdf(scene_tsdf_name);

      std::string scene_keypoints_name = volume_name + "_pts.txt";
      save_volume_keypoints(scene_keypoints_name);

      std::string scene_extrinsics_name = volume_name + "_ext.txt";
      save_volume_to_world_matrix(scene_extrinsics_name, extrinsic_poses, first_frame);

      // Init new volume
      first_frame = curr_frame;
      if (!(curr_frame == total_files - 1))
        curr_frame = curr_frame - 1;
      std::cout << "Creating new volume." << std::endl;
      memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
      voxel_volume.keypoints.clear();
      for (int i = 0; i < 512 * 512 * 1024; i++)
        voxel_volume.tsdf[i] = 1.0f;

    }

  }

}

void generate_data(const std::string &sequence_prefix, const std::string &local_dir) {

  std::vector<std::string> image_list;
  std::vector<std::string> depth_list;
  std::vector<extrinsic> extrinsic_poses;

/////////////////////////////////////////////////////////////////////

  // // Retrieve local directory
  // std::string sequence_name = "hotel_umd/maryland_hotel3/";
  // passwd* pw = getpwuid(getuid());
  // std::string home_dir(pw->pw_dir);
  // std::string local_dir = home_dir + "/marvin/kinfu/data/sun3d/" + sequence_name;

  // // Query RGB-D sequence from SUN3D
  // get_sun3d_data(sequence_name, local_dir, image_list, depth_list, extrinsic_poses);

/////////////////////////////////////////////////////////////////////

  // Retrieve local directory
  // std::string sequence_name = "fire/seq-03/";
  // std::string sequence_prefix = "/data/04/andyz/kinfu/train/fire_seq03/";
  // sys_command("mkdir -p /data/04/andyz/kinfu/train");
  sys_command("mkdir -p " + sequence_prefix);
  // passwd* pw = getpwuid(getuid());
  // std::string home_dir(pw->pw_dir);
  std::cout << std::endl << local_dir << std::endl;
  // std::string local_dir = home_dir + "/marvin/kinfu/data/sevenscenes/" + sequence_name;
  // std::string local_dir = "/data/04/andyz/kinfu/data/sevenscenes/fire/seq-03/";

  // Query RGB-D sequence from
  get_sevenscenes_data(local_dir, image_list, depth_list, extrinsic_poses);

/////////////////////////////////////////////////////////////////////

  // Count total number of frames
  int total_files = 0;
  image_list.size() < depth_list.size() ? total_files = image_list.size() : total_files = depth_list.size();

  // Init voxel volume params
  init_voxel_volume();

  // Init intrinsic matrix K
  std::string local_camera = local_dir + "intrinsics.txt";
  if (std::ifstream(local_camera))
    get_intrinsic_matrix(local_camera);
  else {
    std::cout << "Intrinsics not found. Loading default matrix." << std::endl;
    cam_K.fx = 585.0f; cam_K.fy = 585.0f; cam_K.cx = 320.0f; cam_K.cy = 240.0f;
  }

  // Set first frame of sequence as base coordinate frame
  int base_frame = 0;
  int first_frame = 0;
  std::vector<int> base_frames;
  base_frames.push_back(base_frame);

  // Fuse frames
  for (int curr_frame = 0; curr_frame < total_files; curr_frame++) {
    std::cout << "Fusing frame " << curr_frame << "...";

    // Load image/depth/extrinsic data for frame curr_frame
    // uchar *image_data = (uchar *) malloc(kImageRows * kImageCols * kImageChannels * sizeof(uchar));
    ushort *depth_data = (ushort *) malloc(kImageRows * kImageCols * sizeof(ushort));
    // get_image_data_sun3d(image_list[curr_frame], image_data);
    // get_depth_data_sun3d(depth_list[curr_frame], depth_data);
    // std::cout << depth_list[curr_frame] << std::endl;
    get_depth_data_sevenscenes(depth_list[curr_frame], depth_data);

    // if (curr_frame == 0) {
    //   int bestValue = 0;
    //   for (int i = 0; i < kImageRows * kImageCols ; i++)
    //     if ((int)depth_data[i] > bestValue)
    //       bestValue = (int)depth_data[i];
    //   std::cout << bestValue << std::endl;
    // }

    // Compute camera Rt between current frame and base frame (saved in cam_R and cam_t)
    // then return view frustum bounds for volume (saved in range_grid)
    float cam_R[9] = {0};
    float cam_t[3] = {0};
    float range_grid[6] = {0};
    compute_frustum_bounds(extrinsic_poses, first_frame, curr_frame, cam_R, cam_t, range_grid);

    // Integrate
    time_t tstart, tend;
    tstart = time(0);
    integrate(depth_data, range_grid, cam_R, cam_t);
    tend = time(0);

    // Clear memory
    // free(image_data);
    free(depth_data);

    // Print time
    std::cout << " done!" << std::endl;
    // std::cout << "Integrating took " << difftime(tend, tstart) << " second(s)." << std::endl;

    // Compute intersection between view frustum and current volume
    float view_occupancy = (range_grid[1] - range_grid[0]) * (range_grid[3] - range_grid[2]) * (range_grid[5] - range_grid[4]) / (512 * 512 * 1024);
    // std::cout << "Intersection: " << 100 * view_occupancy << "%" << std::endl;

    if ((curr_frame - first_frame >= 30) || curr_frame == total_files - 1) {

      // Find keypoints
      find_keypoints(range_grid);

      // Save curr volume to file
      std::string volume_name = sequence_prefix + "scene" +  std::to_string(first_frame) + "_" + std::to_string(curr_frame);

      std::string scene_ply_name = volume_name + ".ply";
      save_volume_to_ply(scene_ply_name);

      std::string scene_tsdf_name = volume_name + ".tsdf";
      save_volume_to_tsdf(scene_tsdf_name);

      std::string scene_keypoints_name = volume_name + "_pts.txt";
      save_volume_keypoints(scene_keypoints_name);

      std::string scene_extrinsics_name = volume_name + "_ext.txt";
      save_volume_to_world_matrix(scene_extrinsics_name, extrinsic_poses, first_frame);


      // // Check for loop closure and set a base frame
      base_frame = curr_frame;
      // float base_frame_intersection = 0;
      // for (int i = 0; i < base_frames.size(); i++) {
      //   float cam_R[9] = {0};
      //   float cam_t[3] = {0};
      //   float range_grid[6] = {0};
      //   compute_frustum_bounds(extrinsic_poses, base_frames[i], curr_frame, cam_R, cam_t, range_grid);
      //   float view_occupancy = (range_grid[1]-range_grid[0])*(range_grid[3]-range_grid[2])*(range_grid[5]-range_grid[4])/(512*512*1024);
      //   if (view_occupancy > std::max(0.7f, base_frame_intersection)) {
      //     base_frame = base_frames[i];
      //     base_frame_intersection = view_occupancy;
      //   }
      // }
      // base_frames.push_back(base_frame);

      // Init new volume
      first_frame = curr_frame;
      if (!(curr_frame == total_files - 1))
        curr_frame = curr_frame - 1;
      std::cout << "Creating new volume." << std::endl;
      memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
      voxel_volume.keypoints.clear();
      for (int i = 0; i < 512 * 512 * 1024; i++)
        voxel_volume.tsdf[i] = 1.0f;

    }

    // std::cout << std::endl;
  }

}

// void generate_data_reloc(const std::string &sensor_file, const std::string &local_dir, int start_idx, int end_idx) {

//   std::vector<std::string> image_list;
//   std::vector<std::string> depth_list;
//   std::vector<extrinsic> extrinsic_poses;

// /////////////////////////////////////////////////////////////////////

//   // Retrieve local directory
//   sys_command("mkdir -p " + local_dir);
//   // std::cout << std::endl << local_dir << std::endl;

//   // Query RGB-D sequence and add extrinsic poses
//   // get_sevenscenes_data(local_dir, image_list, depth_list, extrinsic_poses);
//   ml::SensorData sensor_data(sensor_file);
//   std::cout << sensor_data << std::endl;
//   for (int i = 0; i < sensor_data.m_frames.size(); i++) {
//     extrinsic m;
//     for (int d = 0; d < 3; d++) {
//       m.R[d * 3 + 0] = sensor_data.m_frames[i].m_frameToWorld.matrix[d * 4 + 0];
//       m.R[d * 3 + 1] = sensor_data.m_frames[i].m_frameToWorld.matrix[d * 4 + 1];
//       m.R[d * 3 + 2] = sensor_data.m_frames[i].m_frameToWorld.matrix[d * 4 + 2];
//       m.t[d] = sensor_data.m_frames[i].m_frameToWorld.matrix[d * 4 + 3];
//     }
//     extrinsic_poses.push_back(m);
//   }

// /////////////////////////////////////////////////////////////////////

//   // Count total number of frames
//   int total_files = sensor_data.m_frames.size();
//   // image_list.size() < depth_list.size() ? total_files = image_list.size() : total_files = depth_list.size();

//   // Init voxel volume params
//   init_voxel_volume();

//   // Init intrinsic matrix K
//   cam_K.fx = 572.0f; cam_K.fy = 572.0f; cam_K.cx = 320.0f; cam_K.cy = 240.0f;
//   // std::string local_camera = local_dir + "intrinsics.txt";
//   // if (std::ifstream(local_camera))
//   //   get_intrinsic_matrix(local_camera);
//   // else {
//   //   std::cout << "Intrinsics not found. Loading default matrix." << std::endl;
//   //   cam_K.fx = 585.0f; cam_K.fy = 585.0f; cam_K.cx = 320.0f; cam_K.cy = 240.0f;
//   // }

//   // Set first frame of sequence as base coordinate frame
//   int first_frame = start_idx;

//   // Fuse frames
//   for (int curr_frame = start_idx; curr_frame <= end_idx; curr_frame++) {
//     std::cout << "Fusing frame " << curr_frame << "...";

//     // Load image/depth/extrinsic data for frame curr_frame
//     // ushort *depth_data = (ushort *) malloc(kImageRows * kImageCols * sizeof(ushort));
//     ushort* depth_data = sensor_data.m_frames[curr_frame].decompressDepthAlloc();
//     // get_depth_data_sevenscenes(depth_list[curr_frame], depth_data);

//     // Compute camera Rt between current frame and base frame (saved in cam_R and cam_t)
//     // then return view frustum bounds for volume (saved in range_grid)
//     float cam_R[9] = {0};
//     float cam_t[3] = {0};
//     float range_grid[6] = {0};
//     compute_frustum_bounds(extrinsic_poses, first_frame, curr_frame, cam_R, cam_t, range_grid);

//     // Integrate
//     integrate(depth_data, range_grid, cam_R, cam_t);

//     // Clear memory
//     // free(image_data);
//     free(depth_data);

//     // Print time
//     std::cout << " done!" << std::endl;
//     // std::cout << "Integrating took " << difftime(tend, tstart) << " second(s)." << std::endl;

//     // Compute intersection between view frustum and current volume
//     float view_occupancy = (range_grid[1] - range_grid[0]) * (range_grid[3] - range_grid[2]) * (range_grid[5] - range_grid[4]) / (512 * 512 * 1024);
//     // std::cout << "Intersection: " << 100 * view_occupancy << "%" << std::endl;

//     if ((curr_frame - first_frame >= 30)) {

//       // Find keypoints
//       find_keypoints(range_grid);

//       // Save curr volume to file
//       std::string volume_name = local_dir + "scene" +  std::to_string(first_frame) + "_" + std::to_string(curr_frame);

//       std::string scene_ply_name = volume_name + ".ply";
//       save_volume_to_ply(scene_ply_name);

//       std::string scene_tsdf_name = volume_name + ".tsdf";
//       save_volume_to_tsdf(scene_tsdf_name);

//       std::string scene_keypoints_name = volume_name + "_pts.txt";
//       save_volume_keypoints(scene_keypoints_name);

//       std::string scene_extrinsics_name = volume_name + "_ext.txt";
//       save_volume_to_world_matrix(scene_extrinsics_name, extrinsic_poses, first_frame);

//       // Init new volume
//       first_frame = curr_frame;
//       if (!(curr_frame == end_idx))
//         curr_frame = curr_frame - 1;
//       std::cout << "Creating new volume." << std::endl;
//       memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
//       voxel_volume.keypoints.clear();
//       for (int i = 0; i < 512 * 512 * 1024; i++)
//         voxel_volume.tsdf[i] = 1.0f;

//     }

//     if (curr_frame == end_idx) {
//       std::cout << "Not enough frames for new volume." << std::endl;
//       break;
//     }


//   }





// }

////////////////////////////////////////////////////////////////////////////////

void checkout_tsdf(const std::string &scene_name, float* tsdf_volume, int x1, int x2, int y1, int y2, int z1, int z2) {
  // Reads in volume data (xyz order float format) specified by indicies (inclusive)
  // TSDF volume file format:
  //     start index of volume 1*(int32)
  //     number of elements (N) in volume 1*(int32)
  //     volume values N*(float)

  std::string filename = scene_name + ".tsdf";

  std::ifstream inFile(filename, std::ios::binary | std::ios::in);
  int offset;
  int num_elements;
  inFile.read((char*)&offset, sizeof(int));
  inFile.read((char*)&num_elements, sizeof(int));
  int end_idx = offset + num_elements - 1;

  int x_dim = x2 - x1 + 1;
  int y_dim = y2 - y1 + 1;
  // int z_dim = z2 - z1 + 1;

  // for (int z = z1; z <= z2; z++) {
  //   for (int y = y1; y <= y2; y++) {
  //     for (int x = x1; x <= x2; x++) {
  //       int volume_idx = z * 512 * 512 + y * 512 + x;
  //       if (volume_idx > end_idx || volume_idx < offset)
  //         tsdf_volume[(z - z1)*y_dim * x_dim + (y - y1)*x_dim + (x - x1)] = 1.0f;
  //       else {
  //         inFile.seekg(8 + sizeof(float) * (volume_idx - offset));
  //         // float tsdf_value;
  //         inFile.read((char*)&tsdf_volume[(z - z1)*y_dim * x_dim + (y - y1)*x_dim + (x - x1)], sizeof(float));
  //         // tsdf_volume[(z-z1)*y_dim*x_dim+(y-y1)*x_dim+(x-x1)] = tsdf_value;
  //       }
  //     }
  //   }
  // }

  for (int z = z1; z <= z2; z++) {
    for (int y = y1; y <= y2; y++) {
      int volume_idx = z * 512 * 512 + y * 512 + x1;
      // If can't load entire x row
      if ((volume_idx - x1 + x2) > end_idx || volume_idx < offset) {
        for (int x = x1; x <= x2; x++) {
          volume_idx = z * 512 * 512 + y * 512 + x;
          if (volume_idx > end_idx || volume_idx < offset)
            tsdf_volume[(z - z1)*y_dim * x_dim + (y - y1)*x_dim + (x - x1)] = 1.0f;
          else {
            inFile.seekg(8 + sizeof(float) * (volume_idx - offset));
            // float tsdf_value;
            inFile.read((char*)&tsdf_volume[(z - z1)*y_dim * x_dim + (y - y1)*x_dim + (x - x1)], sizeof(float));
            // tsdf_volume[(z-z1)*y_dim*x_dim+(y-y1)*x_dim+(x-x1)] = tsdf_value;
          }
        }
        // If can load entire x row
      } else {
        inFile.seekg(8 + sizeof(float) * (volume_idx - offset));
        inFile.read((char*)&tsdf_volume[(z - z1)*y_dim * x_dim + (y - y1)*x_dim], x_dim * sizeof(float));
      }
    }
  }

  inFile.close();

}

////////////////////////////////////////////////////////////////////////////////

void checkout_ext(const std::string &scene_name, float* ext_mat) {

  std::string filename = scene_name + "_ext.txt";

  int iret;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < 16; i++)
    iret = fscanf(fp, "%f", &ext_mat[i]);
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void checkout_keypts(const std::string &scene_name, std::vector<keypoint> &keypoints) {

  std::string filename = scene_name + "_pts.txt";

  FILE *fp = fopen(filename.c_str(), "r");
  int iret;
  float num_points;
  iret = fscanf(fp, "%f", &num_points);
  for (int i = 0; i < num_points; i++) {
    keypoint tmp_keypt;
    iret = fscanf(fp, "%f", &tmp_keypt.x);
    iret = fscanf(fp, "%f", &tmp_keypt.y);
    iret = fscanf(fp, "%f", &tmp_keypt.z);
    keypoints.push_back(tmp_keypt);
  }
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void grid2world(keypoint keypt_grid, float* ext_mat, keypoint *keypt_world, bool is_7scenes) {

  float sx = (float)keypt_grid.x * 0.01;
  float sy = (float)keypt_grid.y * 0.01;
  float sz = (float)keypt_grid.z * 0.01;

  if (is_7scenes) {
    sx = ((float)keypt_grid.x + 1) * 0.01 - 512 * 0.01 / 2;
    sy = ((float)keypt_grid.y + 1) * 0.01 - 512 * 0.01 / 2;
    sz = ((float)keypt_grid.z + 1) * 0.01 - 0.5;
  }

  // transform
  keypt_world->x = ext_mat[0] * sx + ext_mat[1] * sy + ext_mat[2] * sz;
  keypt_world->y = ext_mat[4] * sx + ext_mat[5] * sy + ext_mat[6] * sz;
  keypt_world->z = ext_mat[8] * sx + ext_mat[9] * sy + ext_mat[10] * sz;

  keypt_world->x = keypt_world->x + ext_mat[3];
  keypt_world->y = keypt_world->y + ext_mat[7];
  keypt_world->z = keypt_world->z + ext_mat[11];
}

////////////////////////////////////////////////////////////////////////////////

void world2grid(keypoint keypt_world, float* ext_mat_inv, keypoint *keypt_grid, bool is_7scenes) {
  keypt_grid->x = ext_mat_inv[0] * keypt_world.x + ext_mat_inv[1] * keypt_world.y + ext_mat_inv[2] * keypt_world.z;
  keypt_grid->y = ext_mat_inv[4] * keypt_world.x + ext_mat_inv[5] * keypt_world.y + ext_mat_inv[6] * keypt_world.z;
  keypt_grid->z = ext_mat_inv[8] * keypt_world.x + ext_mat_inv[9] * keypt_world.y + ext_mat_inv[10] * keypt_world.z;
  keypt_grid->x = keypt_grid->x + ext_mat_inv[3];
  keypt_grid->y = keypt_grid->y + ext_mat_inv[7];
  keypt_grid->z = keypt_grid->z + ext_mat_inv[11];

  if (is_7scenes) {
    keypt_grid->x = round(((((float)(keypt_grid->x)) + 512 * 0.01 / 2) / 0.01) - 1);
    keypt_grid->y = round(((((float)(keypt_grid->y)) + 512 * 0.01 / 2) / 0.01) - 1);
    keypt_grid->z = round(((((float)(keypt_grid->z)) + 0.5) / 0.01) - 1);
  } else {
    keypt_grid->x = round((keypt_grid->x) / 0.01);
    keypt_grid->y = round((keypt_grid->y) / 0.01);
    keypt_grid->z = round((keypt_grid->z) / 0.01);
  }
}

////////////////////////////////////////////////////////////////////////////////

void tsdf2ply(const std::string &filename, float* scene_tsdf, float tsdf_threshold, float* ext_mat, int x_dim, int y_dim, int z_dim, bool use_ext, keypoint pc_color) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < x_dim * y_dim * z_dim; i++)
    if (std::abs(scene_tsdf[i]) < tsdf_threshold)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
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

  // Create point cloud content for ply file
  for (int z = 0; z < z_dim; z++) {
    for (int y = 0; y < y_dim; y++) {
      for (int x = 0; x < x_dim; x++) {
        if (std::abs(scene_tsdf[z * y_dim * x_dim + y * x_dim + x]) < tsdf_threshold) {

          // grid to world coords (7scene)
          // float sx = ((float)x + 1) * 0.01 - 512 * 0.01 / 2;
          // float sy = ((float)y + 1) * 0.01 - 512 * 0.01 / 2;
          // float sz = ((float)z + 1) * 0.01 - 0.5;

          // (synth)
          // float sx = (float)x * 0.01;
          // float sy = (float)y * 0.01;
          // float sz = (float)z * 0.01;

          if (use_ext) {
            float sx = (float)x;
            float sy = (float)y;
            float sz = (float)z;
            float fx = ext_mat[0] * sx + ext_mat[1] * sy + ext_mat[2] * sz;
            float fy = ext_mat[4] * sx + ext_mat[5] * sy + ext_mat[6] * sz;
            float fz = ext_mat[8] * sx + ext_mat[9] * sy + ext_mat[10] * sz;
            fx = fx + ext_mat[3];
            fy = fy + ext_mat[7];
            fz = fz + ext_mat[11];
            fwrite(&fx, sizeof(float), 1, fp);
            fwrite(&fy, sizeof(float), 1, fp);
            fwrite(&fz, sizeof(float), 1, fp);

            uchar r = (uchar) pc_color.x;
            uchar g = (uchar) pc_color.y;
            uchar b = (uchar) pc_color.z;
            fwrite(&r, sizeof(uchar), 1, fp);
            fwrite(&g, sizeof(uchar), 1, fp);
            fwrite(&b, sizeof(uchar), 1, fp);
          } else {
            float sx = (float)x;
            float sy = (float)y;
            float sz = (float)z;
            fwrite(&sx, sizeof(float), 1, fp);
            fwrite(&sy, sizeof(float), 1, fp);
            fwrite(&sz, sizeof(float), 1, fp);

            uchar r = (uchar) pc_color.x;
            uchar g = (uchar) pc_color.y;
            uchar b = (uchar) pc_color.z;
            fwrite(&r, sizeof(uchar), 1, fp);
            fwrite(&g, sizeof(uchar), 1, fp);
            fwrite(&b, sizeof(uchar), 1, fp);
          }


          // transform
          // float fx = ext_mat[0] * sx + ext_mat[1] * sy + ext_mat[2] * sz;
          // float fy = ext_mat[4] * sx + ext_mat[5] * sy + ext_mat[6] * sz;
          // float fz = ext_mat[8] * sx + ext_mat[9] * sy + ext_mat[10] * sz;
          // fx = fx + ext_mat[3];
          // fy = fy + ext_mat[7];
          // fz = fz + ext_mat[11];
          // fwrite(&fx, sizeof(float), 1, fp);
          // fwrite(&fy, sizeof(float), 1, fp);
          // fwrite(&fz, sizeof(float), 1, fp);
        }
      }
    }
  }
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void tsdfpt2ply(const std::string &filename, float* scene_tsdf, float tsdf_threshold, keypoint keypt, int x_dim, int y_dim, int z_dim) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < x_dim * y_dim * z_dim; i++)
    if (std::abs(scene_tsdf[i]) < tsdf_threshold)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points + 356 + 20);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int z = 0; z < z_dim; z++) {
    for (int y = 0; y < y_dim; y++) {
      for (int x = 0; x < x_dim; x++) {

        if (std::abs(scene_tsdf[z * y_dim * x_dim + y * x_dim + x]) < tsdf_threshold) {
          float fx = (float)x;
          float fy = (float)y;
          float fz = (float)z;
          fwrite(&fx, sizeof(float), 1, fp);
          fwrite(&fy, sizeof(float), 1, fp);
          fwrite(&fz, sizeof(float), 1, fp);
          uchar r = (uchar)180;
          uchar g = (uchar)180;
          uchar b = (uchar)180;
          fwrite(&r, sizeof(uchar), 1, fp);
          fwrite(&g, sizeof(uchar), 1, fp);
          fwrite(&b, sizeof(uchar), 1, fp);
        }

        if (x == keypt.x && y == keypt.y && z == keypt.z) {
          for (int k = -1; k <= 1; k++) {
            for (int j = -1; j <= 1; j++) {
              for (int i = -1; i <= 1; i++) {
                int num_edges = 0;
                if (k == -1 || k == 1)
                  num_edges++;
                if (j == -1 || j == 1)
                  num_edges++;
                if (i == -1 || i == 1)
                  num_edges++;
                if (num_edges >= 2) {
                  float float_x = (float) x + i;
                  float float_y = (float) y + j;
                  float float_z = (float) z + k;
                  fwrite(&float_x, sizeof(float), 1, fp);
                  fwrite(&float_y, sizeof(float), 1, fp);
                  fwrite(&float_z, sizeof(float), 1, fp);
                  uchar r = (uchar)255;
                  uchar g = (uchar)0;
                  uchar b = (uchar)0;
                  fwrite(&r, sizeof(uchar), 1, fp);
                  fwrite(&g, sizeof(uchar), 1, fp);
                  fwrite(&b, sizeof(uchar), 1, fp);
                }
              }
            }
          }

          for (int k = -15; k <= 15; k++) {
            for (int j = -15; j <= 15; j++) {
              for (int i = -15; i <= 15; i++) {
                int num_edges = 0;
                if (k == -15 || k == 15)
                  num_edges++;
                if (j == -15 || j == 15)
                  num_edges++;
                if (i == -15 || i == 15)
                  num_edges++;
                if (num_edges >= 2) {
                  float float_x = (float) x + i;
                  float float_y = (float) y + j;
                  float float_z = (float) z + k;
                  fwrite(&float_x, sizeof(float), 1, fp);
                  fwrite(&float_y, sizeof(float), 1, fp);
                  fwrite(&float_z, sizeof(float), 1, fp);
                  uchar r = (uchar)255;
                  uchar g = (uchar)0;
                  uchar b = (uchar)0;
                  fwrite(&r, sizeof(uchar), 1, fp);
                  fwrite(&g, sizeof(uchar), 1, fp);
                  fwrite(&b, sizeof(uchar), 1, fp);
                }
              }
            }
          }
        }
      }
    }
  }
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void checkout_data(const std::string &scene_name, keypoint keypt) {
  // keypoint file format:
  //     number of keypoints (N) 1*(int32)
  //     keypoint xyz 3*N*(float)

  // voxel-to-world extrinsic file format:
  //     row-major matrix 16*(float)

  // Read tsdf volume
  float *scene_tsdf = new float[512 * 512 * 1024];
  checkout_tsdf(scene_name, scene_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);

  // float *ext_mat = new float[16];
  // checkout_ext(scene_name, ext_mat);

  std::string ply_filename = "test_" + scene_name + ".ply";
  float tsdf_threshold = 0.2;
  int x_dim = 512; int y_dim = 512; int z_dim = 1024;
  // tsdf2ply(ply_filename, scene_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);

  tsdfpt2ply(ply_filename, scene_tsdf, tsdf_threshold, keypt, x_dim, y_dim, z_dim);

  delete [] scene_tsdf;
}

////////////////////////////////////////////////////////////////////////////////

// float gen_random_float(float min, float max) {
//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<double> dist(min, max - 0.0001);
//   return dist(mt);
// }

////////////////////////////////////////////////////////////////////////////////

void get_files_in_directory(const std::string &directory, std::vector<std::string> &file_list, const std::string &match_substr) {
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (directory.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string filename(ent->d_name);
      if (filename.find(match_substr) != std::string::npos && filename != "." && filename != "..")
        file_list.push_back(filename);
    }
    closedir (dir);
  } else {
    perror ("Could not look into training directory!");
  }
}

////////////////////////////////////////////////////////////////////////////////

void create_match_nonmatch(std::string data_directory, float *volume_o_tsdf, float *volume_m_tsdf, float *volume_nm_tsdf, int volume_iter) {
  bool success = false;
  while (!success) {
    // Look in training directory for sequence folders
    // std::string data_directory = "train";
    std::vector<std::string> sequence_names;
    std::string dir_match_str = "";
    get_files_in_directory(data_directory, sequence_names, dir_match_str);

    // Pick two random sequences (one for match, one for non-match)
    int rand_sequence_i_m = (int) floor(gen_random_float(0.0, sequence_names.size()));
    int rand_sequence_i_nm = (int) floor(gen_random_float(0.0, sequence_names.size()));
    std::string sequence_dir_m = data_directory + "/" + sequence_names[rand_sequence_i_m];
    std::string sequence_dir_nm = data_directory + "/" + sequence_names[rand_sequence_i_nm];

    // Look into sequence directories for .tsdf files
    std::vector<std::string> scene_names_o;
    std::vector<std::string> scene_names_nm;
    std::string tsdf_match_string = ".tsdf";
    get_files_in_directory(sequence_dir_m, scene_names_o, tsdf_match_string);
    get_files_in_directory(sequence_dir_nm, scene_names_nm, tsdf_match_string);

    // Pick a random scene from match sequence
    int rand_scene_i_o = (int) floor(gen_random_float(0.0, scene_names_o.size()));
    bool is_7scenes = scene_names_o[rand_scene_i_o].find("scene") != std::string::npos;
    std::string scene_dir_o = sequence_dir_m + "/" + scene_names_o[rand_scene_i_o];
    scene_dir_o = scene_dir_o.substr(0, scene_dir_o.length() - tsdf_match_string.length());

    // Extract this scene's extrinsics
    float *ext_mat_o = new float[16];
    checkout_ext(scene_dir_o, ext_mat_o);

    // In this scene, pick a random keypoint
    std::vector<keypoint> keypoints_o;
    checkout_keypts(scene_dir_o, keypoints_o);
    if (keypoints_o.size() == 0)
      continue;
    int rand_keypoint_i_o = (int) floor(gen_random_float(0.0, keypoints_o.size()));
    // int rand_keypoint_i_o = 0;
    keypoint keypt_grid_o = keypoints_o[rand_keypoint_i_o];

    // Convert random keypoint to world coordinates
    keypoint keypt_world_o;
    grid2world(keypt_grid_o, ext_mat_o, &keypt_world_o, is_7scenes);

    // std::cout << scene_dir_o << " " << keypt_grid_o.x << " " << keypt_grid_o.y << " " << keypt_grid_o.z << "| " << keypt_world_o.x << " " << keypt_world_o.y << " " << keypt_world_o.z << std::endl;

    // Search for all possible suitable matches from same sequence (similar sequences if 7scene)
    std::vector<std::string> scene_dirs_m;
    std::vector<keypoint> mapped_keypoints_m;
    std::vector<keypoint> nearby_keypoints_m;
    std::vector<std::string> all_scene_names;

    // // std::cout << sequence_dir_m << std::endl;
    // if (sequence_dir_m.find("_seq") != std::string::npos) {
    //   std::string scene_name_prefix = sequence_names[rand_sequence_i_m].substr(0,sequence_names[rand_sequence_i_m].find("_seq"));
    //   // std::cout << scene_name_prefix << std::endl;
    //   for (std::string &tmp_scene_name : sequence_names) {
    //     if (tmp_scene_name.find(scene_name_prefix) != std::string::npos) {
    //       std::vector<std::string> tmp_scene_names_list;
    //       std::string tsdf_match_string = ".tsdf";
    //       std::string tmp_same_sequence_dir = data_directory + "/" + tmp_scene_name;
    //       get_files_in_directory(tmp_same_sequence_dir, tmp_scene_names_list, tsdf_match_string);
    //       for (std::string &tmp_scene_names_list_iter : tmp_scene_names_list) {
    //         all_scene_names.push_back(tmp_same_sequence_dir + "/" + tmp_scene_names_list_iter);
    //       }
    //     }
    //   }
    // }
    // else {
    //   for (std::string &tmp_scene_name : scene_names_o) {
    //     std::string tmp_scene_dir_m = sequence_dir_m + "/" + tmp_scene_name;
    //     all_scene_names.push_back(tmp_scene_dir_m);
    //   }
    // }

    for (std::string &tmp_scene_name : scene_names_o) {
      std::string tmp_scene_dir_m = sequence_dir_m + "/" + tmp_scene_name;
      all_scene_names.push_back(tmp_scene_dir_m);
    }




    for (std::string &tmp_scene_dir_m : all_scene_names) {
      // Get scene name
      // std::string tmp_scene_dir_m = sequence_dir_m + "/" + tmp_scene_name;
      is_7scenes = tmp_scene_dir_m.find("/scene") != std::string::npos;
      tmp_scene_dir_m = tmp_scene_dir_m.substr(0, tmp_scene_dir_m.length() - tsdf_match_string.length());
      // std::cout << tmp_scene_dir_m << std::endl;

      // Convert base match keypoint to grid coordinates of tmp scene
      keypoint tmp_keypt_grid_m;
      float *tmp_ext_mat_m = new float[16];
      float *tmp_ext_mat_inv_m = new float[16];
      checkout_ext(tmp_scene_dir_m, tmp_ext_mat_m);
      invMatrix(tmp_ext_mat_m, tmp_ext_mat_inv_m);
      world2grid(keypt_world_o, tmp_ext_mat_inv_m, &tmp_keypt_grid_m, is_7scenes);

      // Check if scene has keypoint near base match keypoint
      std::vector<keypoint> tmp_keypoints_m;
      checkout_keypts(tmp_scene_dir_m, tmp_keypoints_m);
      bool keypt_has_match = false;
      for (keypoint &tmp_keypt : tmp_keypoints_m) {
        if (((tmp_keypt.x - tmp_keypt_grid_m.x) * (tmp_keypt.x - tmp_keypt_grid_m.x) + (tmp_keypt.y - tmp_keypt_grid_m.y) * (tmp_keypt.y - tmp_keypt_grid_m.y) + (tmp_keypt.z - tmp_keypt_grid_m.z) * (tmp_keypt.z - tmp_keypt_grid_m.z)) < 25) {
          keypt_has_match = true;
          nearby_keypoints_m.push_back(tmp_keypt);
          break;
        }
      }
      if (keypt_has_match && tmp_scene_dir_m != scene_dir_o) {
        scene_dirs_m.push_back(tmp_scene_dir_m);
        mapped_keypoints_m.push_back(tmp_keypt_grid_m);
        // std::cout << "    " << tmp_scene_dir_m << " " << tmp_keypt_grid_m.x << " " << tmp_keypt_grid_m.y << " " << tmp_keypt_grid_m.z << std::endl;
      }
      delete [] tmp_ext_mat_m;
      delete [] tmp_ext_mat_inv_m;
    }

    // If number of possible matches (not including itself) == 0, redo
    if (scene_dirs_m.size() == 0)
      continue;

    // Pick match
    int rand_scene_i_m = (int) floor(gen_random_float(0.0, scene_dirs_m.size()));
    std::string scene_dir_m = scene_dirs_m[rand_scene_i_m];
    keypoint keypt_grid_m = mapped_keypoints_m[rand_scene_i_m];
    // std::cout << scene_dir_m << " " << keypt_grid_m.x << " " << keypt_grid_m.y << " " << keypt_grid_m.z << std::endl;

    // std::cout << std::endl;

    // Pick a random scene from non-match sequence
    int rand_scene_i_nm = (int) floor(gen_random_float(0.0, scene_names_nm.size()));
    std::string scene_dir_nm = sequence_dir_nm + "/" + scene_names_nm[rand_scene_i_nm];
    scene_dir_nm = scene_dir_nm.substr(0, scene_dir_nm.length() - tsdf_match_string.length());

    // In this scene, pick a random keypoint
    std::vector<keypoint> keypoints_nm;
    checkout_keypts(scene_dir_nm, keypoints_nm);
    if (keypoints_nm.size() == 0)
      continue;
    int rand_keypoint_i_nm = (int) floor(gen_random_float(0.0, keypoints_nm.size()));

    // If in the same volume, make sure it doesn't match base keypoint
    if (scene_dir_o == scene_dir_nm)
      while (rand_keypoint_i_nm == rand_keypoint_i_o)
        rand_keypoint_i_nm = (int) floor(gen_random_float(0.0, keypoints_nm.size()));
    keypoint keypt_grid_nm = keypoints_nm[rand_keypoint_i_nm];

    // If in a different volume but in same sequence, make sure it isn't in set of keypoints nearby possible matches
    if (sequence_dir_m == sequence_dir_nm) {
      for (int tmp_scene_i = 0; tmp_scene_i < scene_dirs_m.size(); tmp_scene_i++)
        if (scene_dirs_m[tmp_scene_i] == scene_dir_nm) {
          keypoint tmp_scene_keypoint = nearby_keypoints_m[tmp_scene_i];
          while (tmp_scene_keypoint.x == keypt_grid_nm.x && tmp_scene_keypoint.y == keypt_grid_nm.y && tmp_scene_keypoint.z == keypt_grid_nm.z) {
            rand_keypoint_i_nm = (int) floor(gen_random_float(0.0, keypoints_nm.size()));
            keypt_grid_nm = keypoints_nm[rand_keypoint_i_nm];
          }
        }
    }

    float tsdf_threshold = 0.2;
    int x_dim = 31; int y_dim = 31; int z_dim = 31;
    float *ext_mat = new float[16];
    memset(ext_mat, 0, sizeof(float) * 16);
    ext_mat[0] = 1;
    ext_mat[5] = 1;
    ext_mat[10] = 1;
    ext_mat[15] = 1;

    checkout_tsdf(scene_dir_o, volume_o_tsdf, keypt_grid_o.x - 15, keypt_grid_o.x + 15, keypt_grid_o.y - 15, keypt_grid_o.y + 15, keypt_grid_o.z - 15, keypt_grid_o.z + 15);
    // tsdf2ply(std::to_string(volume_iter) + "_test_o.ply", volume_o_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);
    checkout_tsdf(scene_dir_m, volume_m_tsdf, keypt_grid_m.x - 15, keypt_grid_m.x + 15, keypt_grid_m.y - 15, keypt_grid_m.y + 15, keypt_grid_m.z - 15, keypt_grid_m.z + 15);
    // tsdf2ply(std::to_string(volume_iter) + "_test_m.ply", volume_m_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);
    checkout_tsdf(scene_dir_nm, volume_nm_tsdf, keypt_grid_nm.x - 15, keypt_grid_nm.x + 15, keypt_grid_nm.y - 15, keypt_grid_nm.y + 15, keypt_grid_nm.z - 15, keypt_grid_nm.z + 15);
    // tsdf2ply(std::to_string(volume_iter) + "_test_nm.ply", volume_nm_tsdf, tsdf_threshold, ext_mat, x_dim, y_dim, z_dim);

    // float *scene_tsdf = new float[512 * 512 * 1024];
    // checkout_tsdf(scene_dir_o, scene_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
    // tsdfpt2ply(std::to_string(volume_iter) + "_test_o.ply", scene_tsdf, tsdf_threshold, keypt_grid_o, 512, 512, 1024);
    // checkout_tsdf(scene_dir_m, scene_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
    // tsdfpt2ply(std::to_string(volume_iter) + "_test_m.ply", scene_tsdf, tsdf_threshold, keypt_grid_m, 512, 512, 1024);
    // checkout_tsdf(scene_dir_nm, scene_tsdf, 0, 512 - 1, 0, 512 - 1, 0, 1024 - 1);
    // tsdfpt2ply(std::to_string(volume_iter) + "_test_nm.ply", scene_tsdf, tsdf_threshold, keypt_grid_nm, 512, 512, 1024);
    // delete [] scene_tsdf;


    // std::cout << std::endl;
    // std::cout << scene_dir_o << " " << keypt_grid_o.x << " " << keypt_grid_o.y << " " << keypt_grid_o.z << std::endl;
    // std::cout << scene_dir_m << " " << keypt_grid_m.x << " " << keypt_grid_m.y << " " << keypt_grid_m.z << std::endl;
    // std::cout << scene_dir_nm << " " << keypt_grid_nm.x << " " << keypt_grid_nm.y << " " << keypt_grid_nm.z << std::endl;

    success = true;

    delete [] ext_mat_o;
    delete [] ext_mat;

  }
}

////////////////////////////////////////////////////////////////////////////////

void generateTestingData() {

  std::string data_dir = "/data/andyz/kinfu/test/sequences";
  std::string test_dir = "/data/andyz/kinfu/test";

  std::cout << data_dir << std::endl;
  std::cout << test_dir << std::endl;

  int num_cases = 10000;

  std::string label_tensor_filename = test_dir + "/labels.tensor";
  std::string data_tensor_filename = test_dir + "/data.tensor";
  std::string label_list_filename = test_dir + "/labels.txt";
  std::string data_list_filename = test_dir + "/data.txt";

  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");
  FILE *data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  FILE *label_list_fp = fopen(label_list_filename.c_str(), "w");
  FILE *data_list_fp = fopen(data_list_filename.c_str(), "w");

  // Write data header
  uint8_t tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  uint32_t tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  uint32_t tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_size = (uint32_t)(num_cases * 2);
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_data_chan = (uint32_t)2;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_volume_dim = (uint32_t)31;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)(num_cases * 2);
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Write list headers
  fprintf(label_list_fp, "%d\n", num_cases * 2);
  fprintf(data_list_fp, "%d %d %d %d\n", num_cases * 2, 31, 31, 31);

  // Create directory to store individual local volume files
  sys_command("mkdir -p " + test_dir + "/volumes");

  for (int i = 0; i < num_cases; i++) {
    float *volume_o_tsdf = new float[31 * 31 * 31];
    float *volume_m_tsdf = new float[31 * 31 * 31];
    float *volume_nm_tsdf = new float[31 * 31 * 31];
    create_match_nonmatch(data_dir, volume_o_tsdf, volume_m_tsdf, volume_nm_tsdf, i);

    // Take absolute value of tsdf
    for (int j = 0; j < 31 * 31 * 31; j++) {
      volume_o_tsdf[j] = std::abs(volume_o_tsdf[j]);
      volume_m_tsdf[j] = std::abs(volume_m_tsdf[j]);
      volume_nm_tsdf[j] = std::abs(volume_nm_tsdf[j]);
    }

    std::string volume_o_filename = test_dir + "/volumes/" + std::to_string(i) + "_o.tsdf";
    std::string volume_m_filename = test_dir + "/volumes/" + std::to_string(i) + "_m.tsdf";
    std::string volume_nm_filename = test_dir + "/volumes/" + std::to_string(i) + "_nm.tsdf";

    std::cout << volume_o_filename << std::endl;
    std::cout << volume_m_filename << std::endl;
    std::cout << volume_nm_filename << std::endl << std::endl;

    // Write local tsdf volume to data tensor file
    fwrite(volume_o_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);
    fwrite(volume_m_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);
    fwrite(volume_o_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);
    fwrite(volume_nm_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);

    // Write ground truth label to label tensor file
    float tmp_label = 1;
    fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);
    tmp_label = 0;
    fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

    // Create individual local tsdf volume binary file
    FILE *tmp_local_volume_o_fp = fopen(volume_o_filename.c_str(), "w");
    fwrite(volume_o_tsdf, sizeof(float), 31 * 31 * 31, tmp_local_volume_o_fp);
    fclose(tmp_local_volume_o_fp);
    FILE *tmp_local_volume_m_fp = fopen(volume_m_filename.c_str(), "w");
    fwrite(volume_m_tsdf, sizeof(float), 31 * 31 * 31, tmp_local_volume_m_fp);
    fclose(tmp_local_volume_m_fp);
    FILE *tmp_local_volume_nm_fp = fopen(volume_nm_filename.c_str(), "w");
    fwrite(volume_nm_tsdf, sizeof(float), 31 * 31 * 31, tmp_local_volume_nm_fp);
    fclose(tmp_local_volume_nm_fp);

    // Write ground truth label and data file location to txt
    fprintf(label_list_fp, "%d\n%d\n", 1, 0);
    fprintf(data_list_fp, "%s %s\n%s %s\n", volume_o_filename.c_str(), volume_m_filename.c_str(), volume_o_filename.c_str(), volume_nm_filename.c_str());

    delete [] volume_o_tsdf;
    delete [] volume_m_tsdf;
    delete [] volume_nm_tsdf;
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);
  fclose(label_list_fp);
  fclose(data_list_fp);

  // FILE *fp = fopen(file_name.c_str(), "w");
  // fprintf(fp, "ply\n");
  // fprintf(fp, "format binary_little_endian 1.0\n");
  // fprintf(fp, "element vertex %d\n", num_points + (int)voxel_volume.keypoints.size() * 20);
  // fprintf(fp, "property float x\n");
  // fprintf(fp, "property float y\n");
  // fprintf(fp, "property float z\n");
  // fprintf(fp, "property uchar red\n");
  // fprintf(fp, "property uchar green\n");
  // fprintf(fp, "property uchar blue\n");
  // fprintf(fp, "end_header\n");

  //     // Convert voxel indices to float, and save coordinates to ply file
  //     float float_x = (float) x;
  //     float float_y = (float) y;
  //     float float_z = (float) z;
  //     fwrite(&float_x, sizeof(float), 1, fp);
  //     fwrite(&float_y, sizeof(float), 1, fp);
  //     fwrite(&float_z, sizeof(float), 1, fp);


}

void tripleD(std::string scene1_dir, std::string scene2_dir, std::vector<keypoint> &keypoints1, std::vector<keypoint> &keypoints2) {

  //-------------------------// Prepare Data //-------------------------//

  int num_cases = keypoints1.size() * keypoints2.size();

  // Init tensor files for marvin
  std::string data_tensor_filename = "/data/andyz/kinfu/sun3d/TMPdata.tensor";
  std::string label_tensor_filename = "/data/andyz/kinfu/sun3d/TMPlabels.tensor";
  FILE *data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

  // Write data header
  uint8_t tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  uint32_t tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  uint32_t tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_data_chan = (uint32_t)2;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_volume_dim = (uint32_t)31;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Extract local volume comparisons and save to tensor files
  for (int i = 0; i < keypoints1.size(); i++) {
    for (int j = 0; j < keypoints2.size(); j++) {
      std::cout << "Iteration " << i*keypoints2.size() + j << "/" << keypoints1.size()*keypoints2.size() - 1 << ": " << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << " | " << keypoints2[j].x << " " << keypoints2[j].y << " " << keypoints2[j].z << std::endl;

      // Extract volume
      float *volume1_tsdf = new float[31 * 31 * 31];
      float *volume2_tsdf = new float[31 * 31 * 31];

      // std::cout << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << std::endl;
      // std::cout << keypoints2[j].x << " " << keypoints2[j].y << " " << keypoints2[j].z << std::endl;

      checkout_tsdf(scene1_dir, volume1_tsdf, keypoints1[i].x - 15, keypoints1[i].x + 15, keypoints1[i].y - 15, keypoints1[i].y + 15, keypoints1[i].z - 15, keypoints1[i].z + 15);
      checkout_tsdf(scene2_dir, volume2_tsdf, keypoints2[j].x - 15, keypoints2[j].x + 15, keypoints2[j].y - 15, keypoints2[j].y + 15, keypoints2[j].z - 15, keypoints2[j].z + 15);

      // Take absolute value of tsdf
      for (int k = 0; k < 31 * 31 * 31; k++) {
        volume1_tsdf[k] = std::abs(volume1_tsdf[k]);
        volume2_tsdf[k] = std::abs(volume2_tsdf[k]);
      }

      // Write local tsdf volume to data tensor file
      fwrite(volume1_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);
      fwrite(volume2_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);

      // Write dummy label to label tensor file
      float tmp_label = 0;
      fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

      delete [] volume1_tsdf;
      delete [] volume2_tsdf;
    }
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  //-------------------------// Run marvin //-------------------------//

  std::string prob_tensor_filename = "prob_response";
  std::string model_filename = "../tripleD/tripleD2_snapshot_450000_testRedKitchen_001_train6_sym.marvin";
  sys_command("rm " + prob_tensor_filename);
  // sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64");
  sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64; ./marvin_sun3d_test test tripleD_sun3d_test.json " + model_filename + " prob " + prob_tensor_filename);
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);

  //-------------------------// Read match scores //-------------------------//

  // std::ifstream inFile("prob_response", std::ios::binary | std::ios::in);
  // int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  // inFile.seekg(size_t(header_bytes));
  // float *labels_raw = new float[num_cases * 2];
  // inFile.read((char*)labels_raw, num_cases * 2 * sizeof(float));
  // inFile.close();
  // float *match_scores = new float[num_cases];
  // for (int i = 0; i < num_cases; i++) {
  //   match_scores[i] = labels_raw[i * 2 + 1];
  //   std::cout << match_scores[i] << std::endl;
  // }

}

void tripleD_modularized(std::string scene1_dir, std::string scene2_dir, std::vector<keypoint> &keypoints1, std::vector<keypoint> &keypoints2) {

  //-------------------------// Prepare Data //-------------------------//

  int num_cases = keypoints1.size() + keypoints2.size();

  // Init tensor files for marvin
  std::string data_tensor_filename = "TMPdata.tensor";
  std::string label_tensor_filename = "TMPlabels.tensor";
  FILE *data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

  // Write data header
  uint8_t tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  uint32_t tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  uint32_t tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_volume_dim = (uint32_t)31;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Extract local volume comparisons and save to tensor files
  for (int i = 0; i < keypoints1.size(); i++) {
    std::cout << "Iteration " << i << "/" << keypoints1.size() + keypoints2.size() - 1 << ": " << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << std::endl;

    // Extract volume
    float *volume_tsdf = new float[31 * 31 * 31];
    checkout_tsdf(scene1_dir, volume_tsdf, keypoints1[i].x - 15, keypoints1[i].x + 15, keypoints1[i].y - 15, keypoints1[i].y + 15, keypoints1[i].z - 15, keypoints1[i].z + 15);

    // Take absolute value of tsdf
    for (int k = 0; k < 31 * 31 * 31; k++) {
      volume_tsdf[k] = std::abs(volume_tsdf[k]);
    }

    // Write local tsdf volume to data tensor file
    fwrite(volume_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);

    // Write dummy label to label tensor file
    float tmp_label = 0;
    fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

    delete [] volume_tsdf;

  }

  for (int i = 0; i < keypoints2.size(); i++) {
    std::cout << "Iteration " << keypoints1.size() + i << "/" << keypoints1.size() + keypoints2.size() - 1 << ": " << keypoints2[i].x << " " << keypoints2[i].y << " " << keypoints2[i].z << std::endl;

    // Extract volume
    float *volume_tsdf = new float[31 * 31 * 31];
    checkout_tsdf(scene2_dir, volume_tsdf, keypoints2[i].x - 15, keypoints2[i].x + 15, keypoints2[i].y - 15, keypoints2[i].y + 15, keypoints2[i].z - 15, keypoints2[i].z + 15);

    // Take absolute value of tsdf
    for (int k = 0; k < 31 * 31 * 31; k++) {
      volume_tsdf[k] = std::abs(volume_tsdf[k]);
    }

    // Write local tsdf volume to data tensor file
    fwrite(volume_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);

    // Write dummy label to label tensor file
    float tmp_label = 0;
    fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

    delete [] volume_tsdf;

  }


  // for (int i = 0; i < keypoints1.size(); i++) {
  //   for (int j = 0; j < keypoints2.size(); j++) {
  //     std::cout << "Iteration " << i*keypoints2.size() + j << "/" << keypoints1.size()*keypoints2.size()-1 << ": " << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << " | " << keypoints2[j].x << " " << keypoints2[j].y << " " << keypoints2[j].z << std::endl;

  //     // Extract volume
  //     float *volume1_tsdf = new float[31 * 31 * 31];
  //     float *volume2_tsdf = new float[31 * 31 * 31];

  //     // std::cout << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << std::endl;
  //     // std::cout << keypoints2[j].x << " " << keypoints2[j].y << " " << keypoints2[j].z << std::endl;

  //     checkout_tsdf(scene1_dir, volume1_tsdf, keypoints1[i].x - 15, keypoints1[i].x + 15, keypoints1[i].y - 15, keypoints1[i].y + 15, keypoints1[i].z - 15, keypoints1[i].z + 15);
  //     checkout_tsdf(scene2_dir, volume2_tsdf, keypoints2[j].x - 15, keypoints2[j].x + 15, keypoints2[j].y - 15, keypoints2[j].y + 15, keypoints2[j].z - 15, keypoints2[j].z + 15);

  //     // Take absolute value of tsdf
  //     for (int k = 0; k < 31 * 31 * 31; k++) {
  //       volume1_tsdf[k] = std::abs(volume1_tsdf[k]);
  //       volume2_tsdf[k] = std::abs(volume2_tsdf[k]);
  //     }

  //     // Write local tsdf volume to data tensor file
  //     fwrite(volume1_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);
  //     fwrite(volume2_tsdf, sizeof(float), 31 * 31 * 31, data_tensor_fp);

  //     // Write dummy label to label tensor file
  //     float tmp_label = 0;
  //     fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

  //     delete [] volume1_tsdf;
  //     delete [] volume2_tsdf;
  //   }
  // }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  //-------------------------// Run marvin to get feature vectors //-------------------------//

  std::string feat_tensor_filename = "feat_response.tensor";
  std::string model_filename = "ddd/dddnet.marvin";
  sys_command("rm " + feat_tensor_filename);
  // sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64");
  sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64; ./marvin_sun3d_test test ddd/featnet.json " + model_filename + " feat " + feat_tensor_filename);
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);

  //-------------------------// Read feature vectors //-------------------------//

  int feat_vect_dim = 2048;
  std::ifstream inFile(feat_tensor_filename, std::ios::binary | std::ios::in);
  int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  inFile.seekg(size_t(header_bytes));
  float *feat_raw = new float[num_cases * feat_vect_dim];
  inFile.read((char*)feat_raw, num_cases * feat_vect_dim * sizeof(float));
  inFile.close();
  // for (int i = 0; i < num_cases * feat_vect_dim; i++) {
  //   std::cout << feat_raw[i] << std::endl;
  // }


  //-------------------------// Concatenate feature vectors //-------------------------//

  num_cases = keypoints1.size() * keypoints2.size();

  // Init tensor files for marvin
  data_tensor_filename = "TMPdata.tensor";
  label_tensor_filename = "TMPlabels.tensor";
  data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

  // Write data header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  tmp_data_chan = (uint32_t)feat_vect_dim * 2;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  for (int i = 0; i < keypoints1.size(); i++) {
    for (int j = 0; j < keypoints2.size(); j++) {
      std::cout << "Iteration " << i*keypoints2.size() + j << "/" << keypoints1.size()*keypoints2.size() - 1 << ": " << keypoints1[i].x << " " << keypoints1[i].y << " " << keypoints1[i].z << " | " << keypoints2[j].x << " " << keypoints2[j].y << " " << keypoints2[j].z << std::endl;

      // float *feat1 = new float[feat_vect_dim];
      // float *feat2 = new float[feat_vect_dim];

      for (int k = 0; k < feat_vect_dim; k++) {
        float feat2_val = feat_raw[(keypoints1.size() + j) * feat_vect_dim + k];
        fwrite(&feat2_val, sizeof(float), 1, data_tensor_fp);
      }

      for (int k = 0; k < feat_vect_dim; k++) {
        float feat1_val = feat_raw[i * feat_vect_dim + k];
        fwrite(&feat1_val, sizeof(float), 1, data_tensor_fp);
      }

      float tmp_label = 0;
      fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

    }
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  //-------------------------// Pass through metric network //-------------------------//


  std::string prob_tensor_filename = "prob_response.tensor";
  model_filename = "ddd/dddnet.marvin";
  sys_command("rm " + prob_tensor_filename);
  // sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64");
  sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64; ./marvin_sun3d_test test ddd/metricnet.json " + model_filename + " prob " + prob_tensor_filename);
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);


  // float *match_scores = new float[num_cases];
  // for (int i = 0; i < num_cases; i++) {
  //   match_scores[i] = feat_raw[i * 2 + 1];
  //   std::cout << match_scores[i] << std::endl;
  // }






  //-------------------------// Read match scores //-------------------------//

  // std::ifstream inFile("prob_response", std::ios::binary | std::ios::in);
  // int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  // inFile.seekg(size_t(header_bytes));
  // float *labels_raw = new float[num_cases * 2];
  // inFile.read((char*)labels_raw, num_cases * 2 * sizeof(float));
  // inFile.close();
  // float *match_scores = new float[num_cases];
  // for (int i = 0; i < num_cases; i++) {
  //   match_scores[i] = labels_raw[i * 2 + 1];
  //   std::cout << match_scores[i] << std::endl;
  // }

}


void fuse_iclnuim(const std::string &sequence_prefix, const std::string &local_dir) {

  std::vector<std::string> image_list;
  std::vector<std::string> depth_list;
  std::vector<extrinsic> extrinsic_poses;

/////////////////////////////////////////////////////////////////////

  // List depth filenames
  std::string depth_match_str = ".png";
  get_files_in_directory(local_dir,  depth_list, depth_match_str);

  // Get extrinsic text file
  std::string ext_match_str = ".txt";
  std::vector<std::string> ext_list;
  get_files_in_directory(local_dir,  ext_list, ext_match_str);

  // Load extrinsics
  FILE *fp = fopen((local_dir + ext_list[0]).c_str(), "r");
  for (int i = 0; i < depth_list.size(); i++) {
    // std::cout << std::endl;
    extrinsic m;
    // float *tmp_ext_mat = new float[16];
    // float *tmp_ext_mat_inv = new float[16];
    int iret; float pose_idx;
    for (int j = 0; j < 3; j++)
      iret = fscanf(fp, "%f", &pose_idx);

    // for (int j = 0; j < 16; j++)
    //   iret = fscanf(fp, "%f", &tmp_ext_mat[j]);





    // for (int d = 0; d < 3; d++) {
    //   m.R[d * 3 + 0] = tmp_ext_mat_inv[d * 4 + 0];
    //   m.R[d * 3 + 1] = tmp_ext_mat_inv[d * 4 + 1];
    //   m.R[d * 3 + 2] = tmp_ext_mat_inv[d * 4 + 2];
    //   m.t[d] = tmp_ext_mat_inv[d * 4 + 3];
    // }
    for (int d = 0; d < 3; ++d) {
      iret = fscanf(fp, "%f", &m.R[d * 3 + 0]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 1]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 2]);
      iret = fscanf(fp, "%f", &m.t[d]);
    }
    for (int j = 0; j < 4; j++)
      iret = fscanf(fp, "%f", &pose_idx);




    extrinsic_poses.push_back(m);
    // for (int i = 0; i < 9; i++)
    //   std::cout << m.R[i] << std::endl;
    // for (int i = 0; i < 3; i++)
    //   std::cout << m.t[i] << std::endl;
    // delete [] tmp_ext_mat;
    // delete [] tmp_ext_mat_inv;
  }

/////////////////////////////////////////////////////////////////////

  // Count total number of frames
  int total_files = depth_list.size();

  // Init voxel volume params
  init_voxel_volume();

  // Init intrinsic matrix K
  cam_K.fx = 481.20f; cam_K.fy = 480.0f; cam_K.cx = 319.50f; cam_K.cy = 239.50f;

  // Set first frame of sequence as base coordinate frame
  int first_frame = 0;

  // Fuse frames
  for (int curr_frame = 0; curr_frame < total_files; curr_frame++) {
    std::cout << "Fusing frame " << curr_frame << "...";

    // Get depth file name
    std::stringstream tmp_depth_filename_ss;
    tmp_depth_filename_ss << std::setw(5) << std::setfill('0') << curr_frame << ".png";
    std::string tmp_depth_filename = local_dir + tmp_depth_filename_ss.str();
    // std::cout << tmp_depth_filename << std::endl;

    // Load image/depth/extrinsic data for frame curr_frame
    // uchar *image_data = (uchar *) malloc(kImageRows * kImageCols * kImageChannels * sizeof(uchar));
    ushort *depth_data = (ushort *) malloc(kImageRows * kImageCols * sizeof(ushort));
    // get_image_data_sun3d(image_list[curr_frame], image_data);
    // get_depth_data_sun3d(depth_list[curr_frame], depth_data);
    // std::cout << depth_list[curr_frame] << std::endl;
    get_depth_data_sevenscenes(tmp_depth_filename, depth_data);

    // for (int j = 0; j < kImageRows * kImageCols; j++) {
    //   float df = ((float)depth_data[j]) / 1000.f;
    //   std::cout << curr_frame << " " << df << std::endl;
    // }

    // if (curr_frame == 0) {
    //   int bestValue = 0;
    //   for (int i = 0; i < kImageRows * kImageCols ; i++)
    //     if ((int)depth_data[i] > bestValue)
    //       bestValue = (int)depth_data[i];
    //   std::cout << bestValue << std::endl;
    // }

    // Compute camera Rt between current frame and base frame (saved in cam_R and cam_t)
    // then return view frustum bounds for volume (saved in range_grid)
    float cam_R[9] = {0};
    float cam_t[3] = {0};
    float range_grid[6] = {0};
    compute_frustum_bounds(extrinsic_poses, first_frame, curr_frame, cam_R, cam_t, range_grid);

    // Integrate
    time_t tstart, tend;
    tstart = time(0);
    integrate(depth_data, range_grid, cam_R, cam_t);
    tend = time(0);

    // Clear memory
    // free(image_data);
    free(depth_data);

    // Print time
    std::cout << " done!" << std::endl;
    // std::cout << "Integrating took " << difftime(tend, tstart) << " second(s)." << std::endl;

    // Compute intersection between view frustum and current volume
    float view_occupancy = (range_grid[1] - range_grid[0]) * (range_grid[3] - range_grid[2]) * (range_grid[5] - range_grid[4]) / (512 * 512 * 1024);
    // std::cout << "Intersection: " << 100 * view_occupancy << "%" << std::endl;

    if ((curr_frame - first_frame >= 30) || curr_frame == total_files - 1) {

      // Find keypoints
      find_keypoints(range_grid);

      // Save curr volume to file
      std::string volume_name = sequence_prefix + "scene" +  std::to_string(first_frame) + "_" + std::to_string(curr_frame);

      std::string scene_ply_name = volume_name + ".ply";
      // save_volume_to_ply(scene_ply_name);
      save_iclnuim_volume_to_ply(scene_ply_name);

      std::string scene_tsdf_name = volume_name + ".tsdf";
      save_volume_to_tsdf(scene_tsdf_name);

      std::string scene_keypoints_name = volume_name + "_pts.txt";
      save_volume_keypoints(scene_keypoints_name);

      std::string scene_extrinsics_name = volume_name + "_ext.txt";
      save_volume_to_world_matrix(scene_extrinsics_name, extrinsic_poses, first_frame);

      // Init new volume
      first_frame = curr_frame;
      if (!(curr_frame == total_files - 1))
        curr_frame = curr_frame - 1;
      std::cout << "Creating new volume." << std::endl;
      memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
      voxel_volume.keypoints.clear();
      for (int i = 0; i < 512 * 512 * 1024; i++)
        voxel_volume.tsdf[i] = 1.0f;

    }

    // std::cout << std::endl;
  }

}

////////////////////////////////////////////////////////////////////////////////

// __global__ void integrate(float* tsdf, float* weight, uchar* image_data, ushort* depth_data, float* range_grid, float* cam_R, float* cam_t) {

//   int volumeIDX = blockIdx.x*blockDim.x + threadIdx.x;

//   if (volumeIDX < 512*512*1024) {
//     tsdf[volumeIDX] = 0.5;
//     weight[volumeIDX] = 0.5;
//   }

//   // int i = blockIdx.x*blockDim.x + threadIdx.x;
//   // if (i < n) y[i] = a*x[i] + y[i];

//   // for (int z = range_grid[2*2+0]; z < range_grid[2*2+1]; z++) {
//   //   for (int y = range_grid[1*2+0]; y < range_grid[1*2+1]; y++) {
//   //     for (int x = range_grid[0*2+0]; x < range_grid[0*2+1]; x++) {

//   //       // grid to world coords
//   //       float tmp_pos[3] = {0};
//   //       tmp_pos[0] = (x + 1)*voxel_volume.unit + voxel_volume.range[0][0];
//   //       tmp_pos[1] = (y + 1)*voxel_volume.unit + voxel_volume.range[1][0];
//   //       tmp_pos[2] = (z + 1)*voxel_volume.unit + voxel_volume.range[2][0];

//   //       // transform
//   //       float tmp_arr[3] = {0};
//   //       tmp_arr[0] = tmp_pos[0] - cam_t[0];
//   //       tmp_arr[1] = tmp_pos[1] - cam_t[1];
//   //       tmp_arr[2] = tmp_pos[2] - cam_t[2];
//   //       tmp_pos[0] = cam_R[0*3+0]*tmp_arr[0] + cam_R[1*3+0]*tmp_arr[1] + cam_R[2*3+0]*tmp_arr[2];
//   //       tmp_pos[1] = cam_R[0*3+1]*tmp_arr[0] + cam_R[1*3+1]*tmp_arr[1] + cam_R[2*3+1]*tmp_arr[2];
//   //       tmp_pos[2] = cam_R[0*3+2]*tmp_arr[0] + cam_R[1*3+2]*tmp_arr[1] + cam_R[2*3+2]*tmp_arr[2];
//   //       if (tmp_pos[2] <= 0)
//   //         continue;

//   //       int px = std::round(cam_K.fx*(tmp_pos[0]/tmp_pos[2]) + cam_K.cx);
//   //       int py = std::round(cam_K.fy*(tmp_pos[1]/tmp_pos[2]) + cam_K.cy);
//   //       if (px < 1 || px > 640 || py < 1 || py > 480)
//   //         continue;

//   //       float p_depth = *(depth_data + (py-1) * kImageCols + (px-1)) / 1000.f;
//   //       if (std::round(p_depth*1000.0f) == 0)
//   //         continue;

//   //       float eta = (p_depth - tmp_pos[2])*sqrt(1 + pow((tmp_pos[0]/tmp_pos[2]), 2) + pow((tmp_pos[1]/tmp_pos[2]), 2));
//   //       if (eta <= -voxel_volume.mu)
//   //         continue;

//   //       // Integrate
//   //       int volumeIDX = z*512*512 + y*512 + x;
//   //       float sdf = std::min(1.0f, eta/voxel_volume.mu);
//   //       float w_old = voxel_volume.weight[volumeIDX];
//   //       float w_new = w_old + 1.0f;
//   //       voxel_volume.weight[volumeIDX] = w_new;
//   //       voxel_volume.tsdf[volumeIDX] = (voxel_volume.tsdf[volumeIDX]*w_old + sdf)/w_new;

//   //     }
//   //   }
//   // }

// }