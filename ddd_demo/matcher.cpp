#include "pc2tsdf/pc2tsdf.h"
#include "detect_keypoints.h"
#include "ddd.h"
#include "fragmentMatcher/fragmentMatcher.h"

using namespace std;

int main()
{
    const float voxelSize = 0.01f;
    const float truncationRadius = 0.05f;
    const float maxKeypointMatchDist = 0.03f;
    const int fragmentCount = 57;
    //const string fragmentPrefix = "/data/andyz/kinfu/data/augICLNUIMDataset/fragments/livingroom1-fragments-ply/cloud_bin_";
    const string fragmentPrefix = "../ddd_data/cloud_bin_";

    vector<string> allFragments;
    
    for (int i = 0; i < fragmentCount; i++)
    {
        const string fragmentFilename = fragmentPrefix + to_string(i) + ".ply";
        if (!util::fileExists(fragmentFilename))
        {
            cout << "file not found: " << fragmentFilename << endl;
            return -1;
        }
        allFragments.push_back(fragmentFilename);
    }
    
    bool dir_success = system("mkdir results");
    for (int i = 0; i < fragmentCount; i++)
    {
        for (int j = 0; j < fragmentCount; j++)
        {
            const string resultFilename = "results/match" + to_string(i) + "-" + to_string(j) + ".txt";
            if (j <= i || util::fileExists(resultFilename))
                continue;
            
            auto result = FragmentMatcher::match(allFragments[i], allFragments[j], i, j, voxelSize, truncationRadius, maxKeypointMatchDist);
            result.saveASCII(resultFilename);
        }
    }

    // for (int i = 0; i < fragmentCount - 1; i++)
    // {
    //     const string resultFilename = "results/match" + to_string(i) + "-" + to_string(i+1) + ".txt";
    //     auto result = FragmentMatcher::match(allFragments[i], allFragments[i+1], i, i+1, voxelSize, truncationRadius, maxKeypointMatchDist);
    //     result.saveASCII(resultFilename);
    // }

    return 0;
}
