import os
import random
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from open_clip import tokenize

# Add necessary directories to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.extend([
    parent_dir,
    os.path.join(os.getcwd(), '../../data-generation'),
    os.path.join(os.getcwd(), '../data-generation'),
    os.path.join(os.getcwd(), '../../data-generation/data_utils'),
    os.path.join(os.getcwd(), '../data-generation/data_utils')
])

from utils import load_dict_from_json, int_keys, how_many_anchors_n_asb, get_image_id_string

class CustomWCDataset(Dataset):
    def __init__(
        self,
        image_paths: str,
        caption_paths: str,
        preprocess_fn: callable,
        num_wc: int,
        dataset_mode: str,
        tokenizer: Any,
        args: Any,
        is_train: bool = True,
        return_raw_captions: bool = False,
        num_alt_options: int = 1,
    ):
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.image_dir = image_paths
        self.return_raw_captions = return_raw_captions
        self.num_wc = num_wc
        self.dataset_mode = dataset_mode
        self.num_alt_options = num_alt_options
        self.is_train = is_train
        self.args = args

        self.index_to_captions = int_keys(load_dict_from_json(caption_paths))
        print(f"Number of captions: {len(self.index_to_captions)} {dataset_mode}")
        self.num_wc_anchors = how_many_anchors_n_asb(self.index_to_captions, self.num_wc)

        if not is_train:
            self.index_to_samples = self._process_captions(self.index_to_captions)

    def _process_captions(self, index_to_captions: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        index_to_samples = {}
        for key, caption_dict in index_to_captions.items():
            sample = {
                'image_id': caption_dict['image_id'],
                'text': caption_dict['simplified_text'],
                'sub-rel-obj': [
                    caption_dict['relation_dict']['subject']['core'],
                    caption_dict['relation_dict']['relation']['core lemma'],
                    caption_dict['relation_dict']['object']['core']
                ]
            }
            if self.dataset_mode == "croco_d":
                sample['b_r_a'] = caption_dict['b_r_a']
            else:
                sample['a_s_b'] = caption_dict['a_s_b']
                
            index_to_samples[key] = sample
        return index_to_samples

    def __len__(self) -> int:
        if self.dataset_mode == "croco_d" or self.dataset_mode == "random" or self.dataset_mode == "vg_relation":
            return len(self.index_to_captions)
        else: 
            return self.num_wc_anchors

    def get_train_item(self, idx):
        current_index = idx % len(self.index_to_captions)
        caption_data = self.index_to_captions[idx]

        image_id = caption_data["image_id"]
        image_path = os.path.join(self.image_dir, get_image_id_string(image_id))

        if image_path is None:
            logging.error(f"Image path not found for image_id: {image_id}")
            raise FileNotFoundError(f"Image path not found for image_id: {image_id}")

        anchor_images = self.preprocess_fn(Image.open(image_path).convert('RGB'))
        anchor_captions_raw = caption_data["simplified_text"]
        anchor_captions = self.tokenizer([str(anchor_captions_raw)])[0]

        wc_images = []
        wc_texts = []
        
        if self.dataset_mode == "croco" or self.dataset_mode == "stative":
            wc_set = caption_data["a_s_b"]
            sampled_wc_rels = random.sample(list(wc_set.keys()), self.num_wc)
            sampled_wc_indices = [random.choice(wc_set[rel]) for rel in sampled_wc_rels]

            for wc_idx in sampled_wc_indices:
                wc_caption_data = self.index_to_captions[wc_idx]
                wc_image_id = wc_caption_data["image_id"]
                wc_image_path = os.path.join(self.image_dir, get_image_id_string(wc_image_id))
                wc_image = self.preprocess_fn(Image.open(wc_image_path).convert('RGB'))
                wc_images.append(wc_image)
                wc_caption_raw = wc_caption_data["simplified_text"]
                wc_caption = self.tokenizer([str(wc_caption_raw)])[0]
                wc_texts.append(wc_caption)
        elif self.dataset_mode == "croco_d":
            b_r_a_idx = caption_data["b_r_a"][0]
            wc_caption_data = self.index_to_captions[b_r_a_idx]
            wc_caption_raw = wc_caption_data["simplified_text"]
            wc_caption = self.tokenizer([str(wc_caption_raw)])[0]
            wc_texts.append(wc_caption)
        
            wc_image_id = wc_caption_data["image_id"]
            wc_image_path = os.path.join(self.image_dir, get_image_id_string(wc_image_id))
            wc_image = self.preprocess_fn(Image.open(wc_image_path).convert('RGB'))
            wc_images.append(wc_image)
        else:
            for _ in range(self.num_wc):
                random_index = random.randint(0, len(self.index_to_captions) - 1)
                random_caption_data = self.index_to_captions[random_index]
                random_image_id = random_caption_data["image_id"]
                random_image_path = os.path.join(self.image_dir, get_image_id_string(random_image_id))

                if random_image_path is None:
                    logging.error(f"Random image path not found for image_id: {random_image_id}")
                    continue

                random_image = self.preprocess_fn(Image.open(random_image_path).convert('RGB'))
                wc_images.append(random_image)
                random_caption_raw = random_caption_data["simplified_text"]
                random_caption = self.tokenizer([str(random_caption_raw)])[0]
                wc_texts.append(random_caption)

        return anchor_images, anchor_captions, wc_images, wc_texts

    def get_val_item(self, index: int) -> Dict[str, List]:
        anchor = self.index_to_samples[index]
        caption_options = [anchor['text']]
        image_id_options = [anchor['image_id']]

        if self.dataset_mode == "croco" or self.dataset_mode == "stative" or self.dataset_mode == "vg_relation":
            self._add_wc_options(anchor, caption_options, image_id_options)
        elif self.dataset_mode == "croco_d":
            self._add_b_r_a_options(anchor, caption_options, image_id_options)
        else:
            self._add_random_options(caption_options, image_id_options, index)

        image_options = self._load_and_preprocess_images(image_id_options)
        caption_options_tokenized = [self.tokenizer(caption) for caption in caption_options]

        if self.return_raw_captions:
            return {"captions": caption_options, "images": image_options}
        else:
            return {
                "image_options": image_options, 
                "caption_options": caption_options_tokenized
            }

    def __getitem__(self, idx):
        if self.is_train:
            return self.get_train_item(idx)
        else:
            return self.get_val_item(idx)

    # Helper methods remain the same
    def _add_wc_options(self, anchor, caption_options, image_id_options):
        wc_relations = anchor['a_s_b']
        chosen_relations = list(wc_relations.keys())[:self.num_wc]
        chosen_wc_indices = [wc_relations[relation][0] for relation in chosen_relations]
        chosen_wc_options = [self.index_to_samples[idx] for idx in chosen_wc_indices]
        caption_options.extend([option['text'] for option in chosen_wc_options])
        image_id_options.extend([option['image_id'] for option in chosen_wc_options])

    def _add_b_r_a_options(self, anchor, caption_options, image_id_options):
        b_r_a_idx = anchor['b_r_a'][0]
        b_r_a_data = self.index_to_samples[b_r_a_idx]
        caption_options.append(b_r_a_data['text'])
        image_id_options.append(b_r_a_data['image_id'])

    def _add_random_options(self, caption_options, image_id_options, current_index):
        next_index = (current_index + 1) % len(self.index_to_samples)
        next_sample = self.index_to_samples[next_index]
        caption_options.append(next_sample['text'])
        image_id_options.append(next_sample['image_id'])

    def _load_and_preprocess_images(self, image_id_options):
        if self.dataset_mode == "vg_relation":
            images = [Image.open(os.path.join(self.image_dir, get_image_id_string(image_id, vg_rel=True, cropped=False))) for image_id in image_id_options]
        else:
            images = [Image.open(os.path.join(self.image_dir, get_image_id_string(image_id, vg_rel=False, cropped=False))) for image_id in image_id_options]
        if self.preprocess_fn:
            images = [self.preprocess_fn(image) for image in images]
        return images

def custom_collate_fn(batch):
    all_images = []
    all_captions = []

    for anchor_images, anchor_captions, wc_images, wc_texts in batch:
        combined_images = [anchor_images] + wc_images
        all_images.extend(combined_images)
        combined_captions = [anchor_captions] + wc_texts
        all_captions.extend(combined_captions)

    all_images = torch.stack(all_images)
    all_captions = torch.stack(all_captions)
    return all_images, all_captions

@dataclass
class DataInfo:
    dataloader: DataLoader

def _get_paths_for_dataset(args, dataset_mode, is_train=True):
    path_mappings = {
        "croco_d": {
            "train": (args.train_images_for_croco_d, args.train_captions_for_croco_d),
            "val": (args.val_images_for_croco_d, args.val_captions_for_croco_d)
        },
        "stative": {
            "train": (args.train_images_for_stative, args.train_captions_for_stative),
            "val": (args.val_images_for_stative, args.val_captions_for_stative)
        },
        "vg_relation": {
            "train": (None, None),
            "val": (args.val_images_for_vg_relation, args.val_captions_for_vg_relation)
        },
        "croco": {
            "train": (args.train_images_for_croco, args.train_captions_for_croco),
            "val": (args.val_images_for_croco, args.val_captions_for_croco)
        },
        "random": {
            "train": (args.train_images_for_croco, args.train_captions_for_croco),
            "val": (args.val_images_for_croco, args.val_captions_for_croco)
        }
    }
    
    mode = "train" if is_train else "val"
    return path_mappings.get(dataset_mode, {}).get(mode, (None, None))

def get_dataset(args, preprocess_fn, tokenizer, dataset_mode, is_train=True, epoch=0):
    image_paths, caption_paths = _get_paths_for_dataset(args, dataset_mode, is_train)
    
    assert image_paths, f"No image paths found for {dataset_mode}"
    assert caption_paths, f"No caption paths found for {dataset_mode}"
    
    dataset = CustomWCDataset(
        image_paths=image_paths,
        caption_paths=caption_paths,
        num_wc=args.num_wc,
        dataset_mode=dataset_mode,
        preprocess_fn=preprocess_fn,
        tokenizer=tokenizer,
        args=args,
        is_train=is_train
    )

    num_samples = len(dataset)
    shuffle = is_train

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=custom_collate_fn if is_train else None
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader)

def get_train_data(args, preprocess_fns, tokenizer=tokenize, epoch=0):
    preprocess_train, _ = preprocess_fns
    data = {}

    dataset_modes = ["croco", "random", "croco_d", "stative"]
    for mode in dataset_modes:
        if _get_paths_for_dataset(args, mode, is_train=True)[0]:
            data_key = f"train_{mode}"
            data[data_key] = get_dataset(
                args=args,
                preprocess_fn=preprocess_train,
                tokenizer=tokenizer,
                dataset_mode=mode,
                is_train=True,
                epoch=epoch
            )

    return data

def get_val_data(args, preprocess_fns, dataset_mode, tokenizer=tokenize, epoch=0):
    _, preprocess_val = preprocess_fns
    data = {}

    dataset_modes = [dataset_mode]
    for mode in dataset_modes:
        if _get_paths_for_dataset(args, mode, is_train=False)[0]:
            data_key = f"val_{mode}"
            data[data_key] = get_dataset(
                args=args,
                preprocess_fn=preprocess_val,
                tokenizer=tokenizer,
                dataset_mode=mode,
                is_train=False,
                epoch=epoch
            )

    return data