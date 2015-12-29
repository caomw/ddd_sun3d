#include <vector>
#include <iostream>

// TSDF Format:
// 3 ints (Xdim,Ydim,Zdim = volume size), followed by float array A of TSDF
// A[z*Ydim*Xdim + y*Xdim + x] = tsdf(x,y,z)

// A demo function on how to read the fragment and create a point coud
void demo_read_fragment_tsdf(const std::string &filename) {

  std::vector<int> dim;
  std::ifstream in(filename, std::ios::binary | std::ios::in);

  // Read TSDF volume dimensions
  for (int i = 0; i < 3; i++) {
    int tmp_dim;
    in.read((char*)&tmp_dim, sizeof(int));
    dim.push_back(tmp_dim);
  }

  // Create TSDF volume
  float* tsdf = new float[dim[0]*dim[1]*dim[2]];

  // Load TSDF values into volume
  for (int i = 0; i < dim[0]*dim[1]*dim[2]; i++)
    in.read((char*)&tsdf[i], sizeof(float));

  in.close();

  //////////////////////////////////////////////////////////////////

  // TSDF threshold for surface points
  float tsdf_threshold = 0.2;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < dim[0]*dim[1]*dim[2]; i++)
    if (std::abs(tsdf[i]) < tsdf_threshold)
      num_points++;

  // Create header for ply file
  std::string pc_filename = ("pc.ply");
  FILE *fp = fopen(pc_filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");

  // Save point cloud
  for (int z = 0; z < dim[2]; z++)
    for (int y = 0; y < dim[1]; y++)
      for (int x = 0; x < dim[0]; x++) {
        int i = z * dim[1] * dim[0] + y * dim[0] + x;
        if (std::abs(tsdf[i]) < tsdf_threshold) {
          float float_x = (float) x;
          float float_y = (float) y;
          float float_z = (float) z;
          fwrite(&float_x, sizeof(float), 1, fp);
          fwrite(&float_y, sizeof(float), 1, fp);
          fwrite(&float_z, sizeof(float), 1, fp);
        }
      }
  fclose(fp);

}

////////////////////////////////////////////////////////////////////////////////

void write_fragment_tsdf(const std::string &filename, float* tsdf, int x_dim, int y_dim, int z_dim) {

  std::ofstream out(filename, std::ios::binary | std::ios::out);

  // Write TSDF volume dimensions
  out.write((char*)&x_dim, sizeof(int));
  out.write((char*)&y_dim, sizeof(int));
  out.write((char*)&z_dim, sizeof(int));

  // Save TSDF values to file
  out.write((char*)tsdf, x_dim*y_dim*z_dim*sizeof(float));

  out.close();

}