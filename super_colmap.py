from .SuperPointDetectors import get_super_points_from_scenes_return
from .matchers import mutual_nn_matcher
import cv2
import os, time
import numpy as np
import argparse
from .database import COLMAPDatabase

camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}

def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 6:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 8:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def init_cameras_database(db, images_path, cameratype, single_camera):
    print("init cameras database ......................................")
    images_name = []
    width = None
    height = None
    for name in sorted(os.listdir(images_path)):
        if 'jpg' in name or 'png' in name:
            images_name.append(name)
            if width is None:
                img = cv2.imread(os.path.join(images_path, name))
                height, width = img.shape[:2]
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)
    if single_camera:
        db.add_camera(cameraModel, width, height, params, camera_id=0)
    for i, name in enumerate(images_name):
        if single_camera:
            db.add_image(name, 0, image_id=i)
            continue
        db.add_camera(cameraModel, width, height, params, camera_id=i)
        db.add_image(name, i, image_id=i)
    return images_name

def import_feature(db, images_path, images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(images_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)

    return sps

def import_feature_from_sps(db, sps, images_name):
    print("feature extraction by super points ...........................")
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)


def match_features(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match
    step_range = [1, 2, 3, 5, 8, 13, 21, 44, 65, 109, 174, 210]
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')
    for step in step_range:
        for i in range(0, num_images - step):
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[i + step]]['descriptors'] * 1.0
            matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
            db.add_matches(i, i + step, matches)
    match_list.close()

def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "database.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    mapper = "colmap mapper --database_path %s --image_path %s --output_path %s" % (
        database_path, images_path, colmap_sparse_path
    )
    operate(mapper)

def geometric_verification(database_path, match_list_path):
    print("Running geometric verification..................................")
    cmd = "colmap matches_importer --database_path %s --match_list_path %s --match_type pairs" % (
        database_path, match_list_path
    )
    operate(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points colmap')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--cameraModel", type=str, required=False, default="SIMPLE_RADIAL")
    parser.add_argument("--images_path", required=False, type=str, default="rgb")
    parser.add_argument("--single_camera", action='store_true')

    args = parser.parse_args()
    database_path = os.path.join(args.projpath, "database.db")
    match_list_path = os.path.join(args.projpath, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    images_path = os.path.join(args.projpath, args.images_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_path, args.cameraModel, args.single_camera)
    sps = import_feature(db, images_path, images_name)
    match_features(db, sps, images_name, match_list_path)
    db.commit()
    db.close()

    geometric_verification(database_path, match_list_path)
    mapper(args.projpath, images_path)