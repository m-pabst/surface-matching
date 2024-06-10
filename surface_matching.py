import cv2
import numpy as np
import open3d as o3d
import time
import json
import os
from numpy import linalg as LA

def load_pcloud_normal_obj(path):
    pcd_obj = o3d.io.read_point_cloud(path)
    # Scaling:
    pcd_obj.scale(0.001, center=(0, 0, 0))
    # Write adapted obj file
    o3d.io.write_point_cloud("data/obj_new.ply", pcd_obj, write_ascii=True, compressed=True)
    return cv2.ppf_match_3d.loadPLYSimple("data/obj_new.ply", 1)

def load_pcloud_normal_testscene(image_path,depth_path):

    color = o3d.io.read_image(image_path)
    depth = o3d.io.read_image(depth_path)

    cam = o3d.camera.PinholeCameraIntrinsic()
    # "10": {"cam_K": [542.0634155273438, 0.0, 306.9838101301975, 0.0, 545.404052734375, 233.993986189671, 0.0, 0.0, 1.0], "depth_scale": 1.0},
    cam.intrinsic_matrix =  [[542, 0.00, 307] , [0.00, 545, 234], [0.00, 0.00, 1.00]]

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)

    pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    pcd_scene.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Write scene file
    if os.path.exists("data/scene.ply"):
        os.remove("data/scene.ply")
    else:
        print("The file does not exist ... Generate 3D Scene")
    o3d.io.write_point_cloud("data/scene.ply", pcd_scene, write_ascii=True, compressed=True)

    return cv2.ppf_match_3d.loadPLYSimple("data/scene.ply", 1)

# point cloud and normal calculation function
image_path = 'data/rgb/000010.png'
depth_path = 'data/depth/000010.png'
pcTest = load_pcloud_normal_testscene(image_path,depth_path) #todo:implement
pc = load_pcloud_normal_obj("data/obj.ply")  #todo:implement

N = 2

# Detect the object in the scene with cv2.ppf_match_3d_PPF3DDetector function
# detector = cv2.ppf_match_3d_PPF3DDetector(param1, param2)

detector = cv2.ppf_match_3d_PPF3DDetector(0.04, 0.1)  

print("start training")
detector.trainModel(pc)

print('start matching')
results = detector.match(pcTest, 0.15, 0.17)

print('start icp')
icp = cv2.ppf_match_3d_ICP(100)

# results contain the refined poses
# results = icp.registerModelToScene(pc, pcTest, results)
_, results = icp.registerModelToScene(pc, pcTest, results[:N])

print("Poses: ")
for i, result in enumerate(results):
    print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
    if i == 0:
        pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
        cv2.ppf_match_3d.writePLY(pct, "pose.ply")
        estim_pose = result.pose

#pct = cv2.ppf_match_3d.transformPCPose(pc, results.pose)
#cv2.ppf_match_3d.writePLY(pct, "pose.ply")

# Calculate the ADD metric for the predicted object pose
# ADD: Average Distance of Model Points
gt_path = "data/scene_gt.json"

# Opening JSON file
gt_data_raw = open(gt_path)
  
# returns JSON object as 
# a dictionary
gt_data = json.load(gt_data_raw)
gt_pose = np.identity(4)
gt_rotation = np.array(gt_data['10'][0]['cam_R_m2c']).reshape((3, 3))
gt_translation = gt_data['10'][0]['cam_t_m2c']

gt_pose[:3,:3] = gt_rotation
gt_pose[:3,3] = gt_translation

#   "10": [{"cam_R_m2c": [0.927282777857, -0.365631457737, 0.0803869788032, -0.105126517996, -0.460406771697, -0.88146260344, 0.359301355006, 0.808914499082, -0.4653646165], 
# "cam_t_m2c": [135.90099999999998, -103.518, 662.0279999999999], "obj_id": 19}],
pc_add = np.delete(pc, [3,4,5], 1)
pc_add = np.insert(pc_add, 3, 1, axis=1)
num_points = len(pc_add) # 5880

model_diameter = None # Needed when scale differes - in my case I preprocessed it 
add = LA.norm(np.subtract(gt_pose @ pc_add.T, estim_pose @ pc_add.T).T) / num_points
print(add) # 8.924970837069385

# visualize the transformed object model with predicted 6d pose in the test scene.
pcd_scene = o3d.io.read_point_cloud("data/scene.ply")
pcd_result = o3d.io.read_point_cloud("pose.ply")
o3d.visualization.draw_geometries([pcd_result, pcd_scene])