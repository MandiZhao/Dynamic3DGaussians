import os 
import json 
import numpy as np
from os.path import join 
from glob import glob 
from natsort import natsorted
from PIL import Image
import shutil
import tqdm 
""" 
1) converts dnerf data format into the dynamic3dgaussians format
NOTE: need to project pointcloud, need to rotate the cam. frame to be compatible with blender

2) converts basketball data format into the dnerf format
"""
 

def dnerf_to_dynaGS(inp_folder, out_folder, folder='train', reference_fname=None, cam_id_offset=0):
    if reference_fname is not None:
        assert '.json' in reference_fname, "reference_fname must be a json file"
        with open(reference_fname) as f:
            reference_data = json.load(f)
        print('reference data keys:', reference_data.keys())
        # data['w'], data['h']: 640, 360
        # data['k']: 150 x 27 x (3x3 matrix): 150 is num of tsteps, 27 is num of cameras
        # 'w2c': 150 x 27 x (4x4 matrix)
        # 'fn': 150 x 27 filenames: '1/000000.jpg'.,...
        # data['cam_id']: 150 x 27

    if folder == 'train':
        assert os.path.exists(join(inp_folder, folder, 'init_pt_cld.npz')), "no init_pt_cld.npz"
        os.makedirs(join(out_folder, 'train'), exist_ok=True)
        shutil.copy(
            join(inp_folder, folder, 'init_pt_cld.npz'), join(out_folder, 'train', 'init_pt_cld.npz'))
    new_meta_data = dict()
    imgs = natsorted(glob(join(inp_folder, folder, "*.png"))) # r_camid_tstep.png 
    tsteps, cam_ids = [], []
    fname_table = dict()
    for img in imgs:
        iname = img.split("/")[-1].split('.')[0]
        t, c = int(iname.split("_")[2]), int(iname.split("_")[1])
        tsteps.append(t)
        cam_ids.append(c)
        assert f"{t}_{c}" not in fname_table, "duplicate fname"
        fname_table[f"{t}_{c}"] = img 
  
    with open(join(inp_folder, f"transforms_{folder}.json")) as f:
        transforms = json.load(f)

    one_img = np.array(Image.open(imgs[0]))
    h, w = one_img.shape[:2]
    new_meta_data['w'] = w
    new_meta_data['h'] = h

    frames = transforms['frames']
    frames_table = dict()
    for f in frames:
        img_fname = f["file_path"].split("/")[-1].split(".")[0]
        t, c = int(img_fname.split("_")[2]), int(img_fname.split("_")[1])
        frames_table[f"{t}_{c}"] = f

    fnames = []
    all_w2c = []
    cam_id = [] 
    k_matrix = [] 
    for t in tqdm.tqdm(sorted(list(set(tsteps)))):
        curr_fnames = []
        curr_w2c = []
        curr_cam_id = []
        curr_k = []
        for c in sorted(list(set(cam_ids))):
            img_fname = fname_table[f"{t}_{c}"]
            # make a copy into new folder
            new_cam_id = c + cam_id_offset
            os.makedirs(
                join(out_folder, 'ims', str(new_cam_id)) , exist_ok=True)
            os.makedirs(
                join(out_folder, 'seg', str(new_cam_id)) , exist_ok=True)
            shutil.copy(
                img_fname, join(out_folder, 'ims', str(new_cam_id), img_fname.split("/")[-1])
                )
            curr_fnames.append(
                join(str(new_cam_id), img_fname.split("/")[-1])
                ) # e.g. "15/r_0_1.png"
        
            c2w = np.array(frames_table[f"{t}_{c}"]["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            w2c[2,:] *= -1
            w2c[1,:] *= -1 # NOTE important!!
            curr_w2c.append(w2c.tolist())
            curr_cam_id.append(
                new_cam_id
            )
            img = Image.open(img_fname) # get the background as seg mask!
            img = np.array(img)
            assert img.shape[-1] == 4, "not RGBA"
            mask = img[:, :, -1]
            mask = Image.fromarray(mask)
            mask.save(join(out_folder, 'seg', str(new_cam_id), img_fname.split("/")[-1]))

            # to get data['k']: use the cam angle and h, w to get intrinsic 
            angle_x, angle_y = transforms["camera_angle_x"], transforms["camera_angle_y"]
            focal_x = (h / 2) / np.tan(angle_x / 2)
            focal_y = (w / 2) / np.tan(angle_y / 2)
            curr_k.append(
                [[focal_x, 0.0, w / 2], [0.0, focal_y, h / 2], [0.0, 0.0, 1.0]]
            )
        
        fnames.append(curr_fnames)
        all_w2c.append(curr_w2c)
        cam_id.append(curr_cam_id)
        k_matrix.append(curr_k)

    print("re-orged fname array shape:", np.array(fnames).shape)
    new_meta_data['fn'] = fnames
    new_meta_data['w2c'] = all_w2c
    new_meta_data['cam_id'] = cam_id
    new_meta_data['k'] = k_matrix

    new_meta_fname = join(out_folder, f"{folder}_meta.json")
    
    with open(new_meta_fname, 'w') as f:
        json.dump(new_meta_data, f, indent=4)

    cam_id_offset += max(cam_ids) + 1 
    return cam_id_offset
    
def dynaGS_to_dnerf(
        inp_folder, 
        out_folder,
        folder='train', 
        time_decimal=3,
        reference_fname="../scene_a_no_depth/transforms_train.json",
        mask_seg=False,
    ):
    if reference_fname is not None:
        with open(reference_fname) as f:
            reference_data = json.load(f)
        print('reference data keys:', reference_data.keys())
        # "camera_angle_x": 0.6911112070083618,
        # "camera_angle_y": 0.6911112070083618,
        # "fl_x": 1111.1110311937682,
        # "fl_y": 1111.1110311937682,
        # "w": 800,
        # "h": 800,
        # "cx": 400.0,
        # "cy": 400.0,
        # "n_frames": 11,
        # "frames": [
        # {"file_path", "time", "transform_matrix": c2w matrix, "type"}
        #]
    old_meta = join(inp_folder, f"{folder}_meta.json")
    with open(old_meta) as f:
        old_meta_data = json.load(f)
    w, h = old_meta_data['w'], old_meta_data['h']
    k_matrix = old_meta_data['k'][0][0] # 3x3 matrix, nested list
    focal_x = k_matrix[0][0] # np.tan(angle_x / 2) = (h / 2) / focal_x
    focal_y = k_matrix[1][1] 
    angle_x = 2 * np.arctan(h / (2 * focal_x))
    angle_y = 2 * np.arctan(w / (2 * focal_y))
    
    new_meta_data = dict(
        camera_angle_x=angle_x,
        camera_angle_y=angle_y,
        fl_x=focal_x,
        fl_y=focal_y,
        w=w, 
        h=h,
        cx=w/2,
        cy=h/2,
    )
    print('begin composing new meta data:', new_meta_data)
    cam_ids = old_meta_data['cam_id'] # 150 tsteps x 27 frames
    w2cs = old_meta_data['w2c'] # 150 tsteps x 27 frames x 4x4 matrix
    fns = old_meta_data['fn'] # 150 tsteps x 27 frames
    # NOTE: dynamicGS has no notion of timestep, so approaximate it from 0 to 1 here
    tot_time = len(cam_ids) - 1
    frames = []
    for t, fnames in enumerate(fns):
        for i, fname in enumerate(fnames):
            cam_id = cam_ids[t][i]
            time = t / tot_time
            w2c = np.array(w2cs[t][i])
            c2w = np.linalg.inv(w2c).tolist()
            frames.append(dict(
                file_path=join(".", folder, "_".join(fname.split("/"))), 
                time=time,
                transform_matrix=c2w,
                type="wrap"
            ))
    new_meta_data["n_frames"] = len(cam_ids)
    new_meta_data["frames"] = frames
    os.makedirs(
        join(out_folder, folder), exist_ok=False)
    new_meta_fname = join(out_folder, f"transforms_{folder}.json")
    with open(new_meta_fname, 'w') as f:
        json.dump(new_meta_data, f, indent=4)
    
    # copy over the images
    for t, fnames in enumerate(fns):
        for i, fname in enumerate(fnames):
            cam_id = cam_ids[t][i] 
            img_fname = join(inp_folder, 'ims', fname)
            # save as png
            # 
            new_fname = join(out_folder, folder, "_".join(fname.split("/")))
            # new_fname = new_fname.replace(".jpg", ".png")
            # img.save(new_fname)
            if mask_seg:
                
                mask_fname = img_fname.replace("ims", "seg")
                mask_fname = mask_fname.replace(".jpg", ".png") # great
                if not os.path.exists(mask_fname):
                    print(f"no mask for {mask_fname}") # NOTE this is weird
                    shutil.copy(img_fname, new_fname)
                    continue
                img = Image.open(img_fname)
                mask = Image.open(mask_fname)
                mask = np.array(mask)  
                masked_img = np.array(img) * mask[..., None]
                # add white background!
                white = mask == 0
                masked_img[white] = 255
                masked_img = Image.fromarray(masked_img)
                masked_img.save(new_fname)
                # breakpoint()
            else:
                shutil.copy(img_fname, new_fname)

    print(f"Done writing to {new_meta_fname}")
    return 
### OPTION1: convert basketball data to dnerf format    
inp_folder = "data/corl_1_dense"
out_folder = "data/corl_1_dense_pano" 
os.makedirs(out_folder, exist_ok=True)
cam_id_offset = 0
for folder in ["train", "test"]:
    cam_id_offset = dnerf_to_dynaGS(inp_folder, out_folder, folder, cam_id_offset=cam_id_offset)
    print(f"cam_id_offset: {cam_id_offset}")

#### OPTION2: convert dnerf data to dynamic3dgaussians format
# inp_folder = "data/basketball"
# out_folder = "../basketball_dnerf"
# for folder in ["train", "test"]:
#     dynaGS_to_dnerf(inp_folder, out_folder, folder, mask_seg=False)
breakpoint()