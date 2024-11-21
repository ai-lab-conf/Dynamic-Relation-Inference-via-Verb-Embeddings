#!/usr/bin/env python3
import os
import sys
print(os.path.join(os.getcwd(), 'training'))
sys.path.append(os.path.join(os.getcwd(), 'training'))
from utils import get_image_id_string, int_keys, load_dict_from_json
import shutil 
from tqdm import tqdm

COCO_IMAGE_DIR = 'coco_images'
COCO_VAL_DIR = os.path.join(COCO_IMAGE_DIR, 'val2017')
COCO_TRAIN_DIR = os.path.join(COCO_IMAGE_DIR, 'train2017')
RAW_CROCO_D_IMAGE_DIR = 'raw_croco_d_images'

DYNAMIC_CROCO_IMAGE_SPLITS = 'dynamic_croco_images'
STATIVE_CROCO_IMAGE_SPLITS = 'stative_croco_images'
CROCO_D_IMAGE_SPLITS = 'croco_d_images'

assert(os.path.exists(COCO_IMAGE_DIR))
assert(os.path.exists(RAW_CROCO_D_IMAGE_DIR))
assert(os.path.exists(COCO_VAL_DIR))
assert(os.path.exists(COCO_TRAIN_DIR))

COCO_TRAIN_IMAGES = set(os.listdir(COCO_TRAIN_DIR))
COCO_VAL_IMAGES = set(os.listdir(COCO_VAL_DIR))
def get_coco_filepath(im_file) -> str:
    """
    Checks in the coco train/val2017 splits and finds the image path associated with the image id.
    """
    #return os.path.join(COCO_IMAGE_DIR, im_file)
    if im_file in COCO_VAL_IMAGES:
        return os.path.join(COCO_VAL_DIR, im_file)
    else:
        #assert(os.path.exists(os.path.join(COCO_VAL_DIR, im_file)))
        return os.path.join(COCO_TRAIN_DIR, im_file)

def make_croco_image_splits(idx_to_caption, label, split_name):
    if label not in ["dynamic", "stative"]:
        raise ValueError("dataset label must be dynamic or stative")
    if split_name not in ["train2017", "test2017"]:
        raise ValueError("split name must be train2017 or test2017")
    
    im_split = DYNAMIC_CROCO_IMAGE_SPLITS if label=="dynamic" else STATIVE_CROCO_IMAGE_SPLITS
    dest_split = os.path.join(im_split, split_name)
    if os.path.exists(dest_split) and len(os.listdir(dest_split))==len(idx_to_caption):
        print(f"Skipping making croco {label} {split_name} because it exists")
        return

    os.makedirs(im_split, exist_ok=True)
    os.makedirs(dest_split, exist_ok=True)
    for caption in tqdm(idx_to_caption.values(), desc=f"creating croco {label} {split_name} image split"):
        image_filename = get_image_id_string(caption['image_id'])
        src_image_path = get_coco_filepath(image_filename)
        dst_image_path = os.path.join(dest_split, image_filename)
        shutil.copy(src_image_path, dst_image_path)
    assert(len(os.listdir(dest_split))==len(idx_to_caption))


def make_croco_d_image_splits(idx_to_caption, split_name):
    if split_name not in ["train2017", "test2017"]:
        raise ValueError("split name must be train2017 or test2017")
    
    
    dest_split = os.path.join(CROCO_D_IMAGE_SPLITS, split_name)
    if os.path.exists(dest_split) and len(os.listdir(dest_split))==len(idx_to_caption):
        print(f"Skipping making croco_d {split_name} because it exists")
        return
    
    os.makedirs(CROCO_D_IMAGE_SPLITS, exist_ok=True)
    os.makedirs(dest_split, exist_ok=True)
    for caption in tqdm(idx_to_caption.values(), desc=f"creating croco_d {split_name} image split"):
        image_filename = get_image_id_string(caption['image_id'])
        if image_filename[:4]=='bra_':
            src_image_path = os.path.join(RAW_CROCO_D_IMAGE_DIR, image_filename)
        else:
            src_image_path = get_coco_filepath(image_filename)
        dst_image_path = os.path.join(dest_split, image_filename)
        shutil.copy(src_image_path, dst_image_path)
    assert(len(os.listdir(dest_split))==len(idx_to_caption))


def make_croco():
    for label in ['dynamic', 'stative']:
        train_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_{label}_train.json')))
        test_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_{label}_test.json')))
        split_to_captions = {
            "train2017": train_idx_to_caption,
            "test2017": test_idx_to_caption
        }
        for split_name, idx_to_caption in split_to_captions.items():
            make_croco_image_splits(idx_to_caption=idx_to_caption,
                                    label=label,
                                    split_name=split_name)
        

def make_croco_d():
    train_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_d_train.json')))
    test_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_d_test.json')))
    split_to_captions = {
            "train2017": train_idx_to_caption,
            "test2017": test_idx_to_caption
    }
    for split_name, idx_to_caption in split_to_captions.items():
        make_croco_d_image_splits(idx_to_caption=idx_to_caption,
                                  split_name=split_name)
        

def sanity_check_helper(split_to_captions, root_im_dir, vg_relation=False):
    for split_name, idx_to_caption in split_to_captions.items():
        split_im_dir = os.path.join(root_im_dir, split_name)
        for caption in idx_to_caption.values():
            assert(os.path.exists(os.path.join(split_im_dir, get_image_id_string(caption['image_id'], vg_rel=vg_relation))))
            assert(os.path.exists(os.path.join(split_im_dir, get_image_id_string(caption['image_id'], vg_rel=vg_relation, cropped=True))))

def sanity_check():
    for label in ['dynamic', 'stative']:
        train_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_{label}_train.json')))
        test_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_{label}_test.json')))
        split_to_captions = {
            "train2017": train_idx_to_caption,
            "test2017": test_idx_to_caption
        }
        sanity_check_helper(split_to_captions, root_im_dir=f"{label}_croco_images")
    
    train_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_d_train.json')))
    test_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', f'croco_d_test.json')))
    split_to_captions = {
            "train2017": train_idx_to_caption,
            "test2017": test_idx_to_caption
    }
    sanity_check_helper(split_to_captions=split_to_captions, 
                        root_im_dir=CROCO_D_IMAGE_SPLITS)
    
    vg_rel_idx_to_caption = int_keys(load_dict_from_json(os.path.join('croco_dataset', 'vg_relation.json')))
    split_to_captions = {
            "vg_relation_images": vg_rel_idx_to_caption
    }
    print("Testing vg relations")
    sanity_check_helper(split_to_captions=split_to_captions, 
                        root_im_dir='.',
                        vg_relation=True)
    print("Passed sanity checks!")


if __name__=='__main__':
    print("Beginnning to make croco image splits...")
    make_croco()
    print("Finished making croco image splits!")
    make_croco_d()
    print("Finished making croco_d image splits")
    print("Deleting unused and redundant images")
    print("Sanity check for each dataset")
    sanity_check()
    print("Removing unused and duplicate images from coco and croco_d")
    shutil.rmtree(COCO_IMAGE_DIR)
    shutil.rmtree(RAW_CROCO_D_IMAGE_DIR)
    print("All done!")