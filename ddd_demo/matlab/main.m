
%% RANSAC
% keypoints1 = dlmread('../TMPkeypoints1.txt',' ');
% keypoints2 = dlmread('../TMPkeypoints2.txt',' ');
% match_idx = dlmread('../TMPmatches.txt',' ');
% matches = [];
% for i=1:size(keypoints1,1)
%     if match_idx(i,2) ~= 0
%         matches = cat(1,matches,[keypoints1(i,:) keypoints2(match_idx(i,2),:)]);
%     end
% end
% error3D_threshold = 0.05;
% [RtRANSAC, inliers] = ransacfitRt([matches(:,1:3)'; matches(:,4:6)'], error3D_threshold, 1);
% dlmwrite('../TMPrt.txt', cat(1, RtRANSAC, [0 0 0 1]), ' ');
% RtRANSAC
% inliers

%% ICP
pointcloud1 = dlmread('../TMPpointcloud1.txt',' ');
pointcloud2 = dlmread('../TMPpointcloud2.txt',' ');
[TR, TT, ER] = icp(pointcloud1',pointcloud2',20,'Matching','kDtree','SmartRejection',2);
dlmwrite('../TMPicpRt.txt', cat(1, cat(2, TR, TT), [0 0 0 1]), ' ');
% pc1 = pointCloud(pointcloud1);
% pc2 = pointCloud((TR*pointcloud2'+repmat(TT,1,size(pointcloud2',2)))');
% pcwrite(pc1,'test1','PLYFormat','binary');
% pcwrite(pc2,'test2','PLYFormat','binary');