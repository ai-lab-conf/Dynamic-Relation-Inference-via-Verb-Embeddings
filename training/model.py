import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import json
import os
import clip
import open_clip
from open_clip import create_model_and_transforms
from open_clip import get_tokenizer as open_clip_tokenizer


def get_model(args: Any, model_name: str, device: str, from_path: Optional[str] = None, root_dir: str = os.getcwd()):
    """Get model and preprocessing functions."""
    # Load from custom path
    if from_path is not None:
        if model_name == "conclip":
            model, preprocess = clip.load("ViT-B/32", device=device)
            checkpoint = torch.load(from_path, map_location = device)
            model = model.float()
            model.load_state_dict(checkpoint['model'])
            return model, preprocess, preprocess, clip.tokenize
        else:
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                "ViT-B-32",
                pretrained=from_path,
                precision=args.precision,
                device=device,
                jit=args.torchscript,  # default = false
                force_quick_gelu=args.force_quick_gelu,  # default = false
                pretrained_image=args.pretrained_image  # default = false
            )
            return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-32')

    #Open_Clip ViT-B-32
    elif model_name == 'open_clip':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-32',
            pretrained=args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,  # default = false
            force_quick_gelu=args.force_quick_gelu,  # default = false
            pretrained_image=args.pretrained_image  # default = false
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-32')

    # elif model_name == 'dac':
    #     model, preprocess_train, preprocess_val = create_model_and_transforms(
    #         'ViT-B-32',
    #         pretrained="/users/mali37/their_models/dac.pt",
    #         precision=args.precision,
    #         device=device,
    #         jit=False,  # default = false
    #         lora = 4,
    #         force_quick_gelu=args.force_quick_gelu,  # default = false
    #         pretrained_image=args.pretrained_image  # default = false
    #     )
    #     return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-32')

     #Open_Clip_ConvNext
    elif model_name == 'Clip_ViT-L-14':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-L-14',
            pretrained='laion2b_s32b_b82k',
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,  
            pretrained_image=args.pretrained_image 
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-L-14')
    elif model_name == 'Clip_g_14':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-g-14',
            pretrained='laion2b_s34b_b88k',
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,  
            pretrained_image=args.pretrained_image 
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-g-14')
    #ConvNext
    elif model_name == 'ConvNext':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'convnext_large_d',
            pretrained='laion2b_s26b_b102k_augreg',
            precision=args.precision,
            device=device,
            jit=args.torchscript,  
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('convnext_large_d')
    elif model_name == 'SigLip':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-16-SigLIP',
            pretrained='webli',
            precision=args.precision,
            device=device,
            jit=args.torchscript,  
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-16-SigLIP')
    #COCO
    elif model_name == 'COCA':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'coca_ViT-B-32',
            pretrained='laion2b_s13b_b90k',
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('coca_ViT-B-32')
    #DFN
    elif model_name == 'DFN':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14-quickgelu',
            pretrained='dfn5b',
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-H-14-quickgelu')
    #EVA
    elif model_name == 'EVA':
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'EVA02-L-14',
            pretrained='merged2b_s4b_b131k',
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('EVA02-L-14')

    # OpenAI CLIP
    elif model_name == 'openai_clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, preprocess
    elif model_name == "ce-clip":
        import gdown
        path = os.path.join(root_dir, "ce-clip.pt")
        if not os.path.exists(path):
            print("Downloading the Ce-CLIP model...")
            gdown.download(id="1DWPw3CtGh5cHz9bW_-iXRSG7BBUVl13K", output=path, quiet=False)
        
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-32', 
            pretrained=path, 
            device=device
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-32')
    
    # NegCLIP
    elif model_name == "negClip":
        import gdown
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-B-32', 
            pretrained=path, 
            device=device
        )
        return model, preprocess_train, preprocess_val, open_clip.get_tokenizer('ViT-B-32')

    raise ValueError(f"Unknown model {model_name}")