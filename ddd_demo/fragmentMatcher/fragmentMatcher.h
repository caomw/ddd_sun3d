
#include "pc2tsdf.h"

class FragmentMatcher
{
public:
    struct KeypointMatch
    {
        vec3f posA;
        vec3f posB;
        float alignmentError;
    };

    struct Result
    {
        void saveASCII(const std::string &filename) const
        {
            std::ofstream file(filename);
            file << "matchCount " << matches.size() << std::endl;
            file << "keypointsACount " << keypointsA.size() << std::endl;
            file << "keypointsBCount " << keypointsB.size() << std::endl;
            file << "transform" << std::endl;
            file << transformBToA;
            
            file << "#matches" << std::endl;
            for (const auto &match : matches)
            {
                file << match.posA << " " << match.posB << " " << match.alignmentError << std::endl;
            }
            file << "#keypointsA" << std::endl;
            for (vec3f v : keypointsA)
                file << v << std::endl;
            file << "#keypointsB" << std::endl;
            for (vec3f v : keypointsB)
                file << v << std::endl;
        }
        bool matchFound;

        mat4f transformBToA;
        
        std::vector<vec3f> keypointsA;
        std::vector<vec3f> keypointsB;
        std::vector<KeypointMatch> matches;
    };

    static void mulMatrix(const float m1[16], const float m2[16], float mOut[16]) {
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

    static float gen_random_float(float min, float max) {
      std::random_device rd;
      std::mt19937 mt(rd());
      std::uniform_real_distribution<double> dist(min, max - 0.0001);
      return dist(mt);
    }

    static void sys_command(std::string str) {
      if (system(str.c_str()))
        return;
    }

    static Result match(const std::string &pointCloudFileA, const std::string &pointCloudFileB, int cloudIndA, int cloudIndB, float voxelSize, float truncationRadius, float maxKeypointMatchDist)
    {
        std::cout << std::endl << "Matching " << pointCloudFileA << " against " << pointCloudFileB << std::endl;
        tic();
        FlatTSDF tsdfA = plyToTSDF(pointCloudFileA, voxelSize, truncationRadius);
        FlatTSDF tsdfB = plyToTSDF(pointCloudFileB, voxelSize, truncationRadius);
        std::cout << "Loading point clouds as TSDFs. ";
        toc();
        // std::cout << "OriginA: " << tsdfA.origin << std::endl;
        // std::cout << "DimA: " << tsdfA.dim << std::endl;

        ///////////////////////////////////////////////////////////////////

        const float k_match_score_thresh = 0.5f;
        const float ransac_k = 3; // RANSAC over top-k > k_match_score_thresh
        const float max_ransac_iter = 500000;
        const float ransac_inlier_thresh = 0.04f;
        
        float* Rt = new float[16]; // Contains rigid transform matrix
        Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;
        align2tsdf(
            tsdfA.data.data(), tsdfA.dim.x, tsdfA.dim.y, tsdfA.dim.z, tsdfA.origin.x, tsdfA.origin.y, tsdfA.origin.z,
            tsdfB.data.data(), tsdfB.dim.x, tsdfB.dim.y, tsdfB.dim.z, tsdfB.origin.x, tsdfB.origin.y, tsdfB.origin.z,
            voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

        ///////////////////////////////////////////////////////////////////

        // TODO: this is redundant with align2tsdf
        Result result;
        result.keypointsA = tsdfA.makeKeypoints();
        result.keypointsB = tsdfB.makeKeypoints();

        result.transformBToA = mat4f::identity();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                result.transformBToA(j, i) = Rt[j * 4 + i];
        
        pc2tsdf::UniformAccelerator acceleratorA(result.keypointsA, maxKeypointMatchDist);
        
        //transform B's keypoints into A's
        std::vector<vec3f> keypointsBtransformed = result.keypointsB;
        for (auto &bPt : keypointsBtransformed)
        {
            const vec3f bPtInA = result.transformBToA * bPt;
            const auto closestPt = acceleratorA.findClosestPoint(bPtInA);
            const float dist = vec3f::dist(bPtInA, closestPt.first);
            if (dist <= maxKeypointMatchDist)
            {
                KeypointMatch match;
                match.posA = closestPt.first;
                match.posB = bPt;
                match.alignmentError = dist;
                result.matches.push_back(match);
            }
        }

        result.matchFound = result.matches.size() > 0;

        std::cout << "Keypoint matches found: " << result.matches.size() << std::endl;

        ///////////////////////////////////////////////////////////////////
        
        float* final_Rt = new float[16];
        final_Rt = Rt;

        // DISABLE ICP FOR NOW (too slow)
        bool use_matlab_icp = false;
        if (use_matlab_icp) {
          tic();
          // Save point clouds to files for matlab to read
          auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
          FILE *fp = fopen("TMPpointcloud1.txt", "w");
          for (int i = 0; i < cloud1.m_points.size(); i++)
            fprintf(fp, "%f %f %f\n", cloud1.m_points[i].x, cloud1.m_points[i].y, cloud1.m_points[i].z);
          fclose(fp);
          auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);
          fp = fopen("TMPpointcloud2.txt", "w");
          for (int i = 0; i < cloud2.m_points.size(); i++) {
            vec3f tmp_point;
            tmp_point.x = Rt[0] * cloud2.m_points[i].x + Rt[1] * cloud2.m_points[i].y + Rt[2] * cloud2.m_points[i].z + Rt[3];
            tmp_point.y = Rt[4] * cloud2.m_points[i].x + Rt[5] * cloud2.m_points[i].y + Rt[6] * cloud2.m_points[i].z + Rt[7];
            tmp_point.z = Rt[8] * cloud2.m_points[i].x + Rt[9] * cloud2.m_points[i].y + Rt[10] * cloud2.m_points[i].z + Rt[11];
            fprintf(fp, "%f %f %f\n", tmp_point.x, tmp_point.y, tmp_point.z);
          }
          fclose(fp);

          // Run matlab ICP
          sys_command("cd matlab; matlab -nojvm < main.m >/dev/null; cd ..");
          float *icp_Rt = new float[16];
          int iret;
          fp = fopen("TMPicpRt.txt", "r");
          for (int i = 0; i < 16; i++) {
            iret = fscanf(fp, "%f", &icp_Rt[i]);
          }
          fclose(fp);

          // Apply ICP Rt to current Rt
          mulMatrix(icp_Rt, Rt, final_Rt);

          delete [] icp_Rt;
          sys_command("rm TMPpointcloud1.txt");
          sys_command("rm TMPpointcloud2.txt");
          sys_command("rm TMPicpRt.txt");

          std::cout << "Using ICP to re-adjust rigid transform. ";
          toc();
        }

        const bool debugDump = true;
        if (debugDump) {
            ///////////////////////////////////////////////////////////////////
            // DEBUG: save point aligned point clouds
            tic();

            auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
            auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);

            // Rotate B points into A using final_Rt
            for (int i = 0; i < cloud2.m_points.size(); i++) {
                vec3f tmp_point;
                tmp_point.x = final_Rt[0] * cloud2.m_points[i].x + final_Rt[1] * cloud2.m_points[i].y + final_Rt[2] * cloud2.m_points[i].z + final_Rt[3];
                tmp_point.y = final_Rt[4] * cloud2.m_points[i].x + final_Rt[5] * cloud2.m_points[i].y + final_Rt[6] * cloud2.m_points[i].z + final_Rt[7];
                tmp_point.z = final_Rt[8] * cloud2.m_points[i].x + final_Rt[9] * cloud2.m_points[i].y + final_Rt[10] * cloud2.m_points[i].z + final_Rt[11];
                cloud2.m_points[i] = tmp_point;
            }

            // Make point clouds colorful
            ml::vec4f color1;
            for (int i = 0; i < 3; i++)
                color1[i] = gen_random_float(0.0, 1.0);
            for (int i = 0; i < cloud1.m_points.size(); i++)
                cloud1.m_colors[i] = color1;
            ml::vec4f color2;
            for (int i = 0; i < 3; i++)
                color2[i] = gen_random_float(0.0, 1.0);
            for (int i = 0; i < cloud2.m_points.size(); i++)
                cloud2.m_colors[i] = color2;

            // Save point clouds to file
            std::string pcfile1 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndA) + ".ply";
            PointCloudIOf::saveToFile(pcfile1, cloud1);
            std::string pcfile2 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndB) + ".ply";
            PointCloudIOf::saveToFile(pcfile2, cloud2);

            std::cout << "Saving point cloud visualizations. ";
            toc();
        }

        ///////////////////////////////////////////////////////////////////

        // // DEBUG: Use MATLAB RANSAC
        // sys_command("cd matlab; matlab -nojvm < main.m; cd ..");
        // float *tmp_matlab_rt = new float[12];
        // int iret;
        // FILE *fp = fopen("TMPrt.txt", "r");
        // for (int i = 0; i < 12; i++) {
        //   iret = fscanf(fp, "%f", &tmp_matlab_rt[i]);
        //   std::cout << tmp_matlab_rt[i] << std::endl;
        // }
        // fclose(fp);

        // if (debugDump) {
        //     // DEBUG: Use MATLAB RANSAC
        //     auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
        //     auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);

        //     // Rotate B points into A using Rt
        //     for (int i = 0; i < cloud2.m_points.size(); i++) {
        //         vec3f tmp_point;
        //         tmp_point.x = tmp_matlab_rt[0] * cloud2.m_points[i].x + tmp_matlab_rt[1] * cloud2.m_points[i].y + tmp_matlab_rt[2] * cloud2.m_points[i].z;
        //         tmp_point.y = tmp_matlab_rt[4] * cloud2.m_points[i].x + tmp_matlab_rt[5] * cloud2.m_points[i].y + tmp_matlab_rt[6] * cloud2.m_points[i].z;
        //         tmp_point.z = tmp_matlab_rt[8] * cloud2.m_points[i].x + tmp_matlab_rt[9] * cloud2.m_points[i].y + tmp_matlab_rt[10] * cloud2.m_points[i].z;
        //         tmp_point.x = tmp_point.x + tmp_matlab_rt[3];
        //         tmp_point.y = tmp_point.y + tmp_matlab_rt[7];
        //         tmp_point.z = tmp_point.z + tmp_matlab_rt[11];
        //         cloud2.m_points[i] = tmp_point;
        //     }

        //     // Make point clouds colorful
        //     ml::vec4f color1;
        //     for (int i = 0; i < 3; i++)
        //         color1[i] = gen_random_float(0.0, 1.0);
        //     for (int i = 0; i < cloud1.m_points.size(); i++)
        //         cloud1.m_colors[i] = color1;
        //     ml::vec4f color2;
        //     for (int i = 0; i < 3; i++)
        //         color2[i] = gen_random_float(0.0, 1.0);
        //     for (int i = 0; i < cloud2.m_points.size(); i++)
        //         cloud2.m_colors[i] = color2;

        //     // Save point clouds to file
        //     std::string pcfile1 = "results/debug_matlab_" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndA) + ".ply";
        //     PointCloudIOf::saveToFile(pcfile1, cloud1);
        //     std::string pcfile2 = "results/debug_matlab_" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndB) + ".ply";
        //     PointCloudIOf::saveToFile(pcfile2, cloud2);
        // }

        return result;
    }

private:
    
    struct FlatTSDF
    {
        std::vector<vec3f> makeKeypoints()
        {
            auto keypointsInt = detect_keypoints_filtered(data.data(), dim.x, dim.y, dim.z);
            std::vector<vec3f> result;
            for (auto &x : keypointsInt)
                result.push_back(vec3f(x[0], x[1], x[2]) * voxelSize + origin);
            return result;
        }
        vec3i dim;
        vec3f origin;
        float voxelSize;
        std::vector<float> data;
    };

    static FlatTSDF plyToTSDF(const std::string &filename, float voxelSize, float truncationRadius)
    {
        auto cloud = PointCloudIOf::loadFromFile(filename);
        std::cout << "Cloud size: " << cloud.m_points.size() << ", p5: " << cloud.m_points[5] << std::endl;
        pc2tsdf::TSDF tsdf;
        pc2tsdf::makeTSDF(cloud, voxelSize, truncationRadius, tsdf);
        
        // Convert TUDF grid data to float array
        FlatTSDF result;
        const vec3i dim = tsdf.data.getDimensions();
        result.dim = dim;
        result.data = std::vector<float>(dim.x * dim.y * dim.z);
        for (float &x : result.data)
            x = 1.0f;
        result.origin = tsdf.worldOrigin;
        result.voxelSize = voxelSize;
        
        for (int z = 0; z < dim.z; z++)
            for (int y = 0; y < dim.y; y++)
                for (int x = 0; x < dim.x; x++)
                    result.data[z * dim.y * dim.x + y * dim.x + x] = tsdf.data(x, y, z) / truncationRadius;
        return result;
    }
};