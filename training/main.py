import os
import gc
import random
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
import clip

# Custom modules
# from training.data import get_train_data
# from training.data import get_val_data
# from training.params import parse_args
# from training.scheduler import cosine_lr
# from training.train import train_one_epoch, evaluate
# from training.model import get_model

from data import get_train_data
from data import get_val_data
from params import parse_args
from scheduler import cosine_lr
from train import train_one_epoch, evaluate
from model import get_model

def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parse_args()
    random_seed(args.seed)

    args.model = args.model.replace('/', '-')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    assert args.precision in ['amp', 'fp32']
    if args.precision == 'fp16':
        print('Warning: It is recommended to use AMP mixed-precision instead of FP16. '
              'FP16 support needs further verification and tuning, especially for training.')

    model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name=args.model, device=device)
    print("Model and preprocess functions loaded.")

    optimizer = None
    scaler = None
    if args.train_captions_for_croco:
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        is_text_p = lambda n: "transformer." in n and not "visual." in n
        is_image_p = lambda n: "visual." in n

        named_parameters = list(model.named_parameters())
        gain_or_bias_params_image = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad and is_image_p(n)]
        rest_params_image = [p for n, p in named_parameters if include(n, p) and p.requires_grad and is_image_p(n)]

        gain_or_bias_params_text = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad and is_text_p(n)]
        rest_params_text = [p for n, p in named_parameters if include(n, p) and p.requires_grad and is_text_p(n)]

        gain_or_bias_params_rest = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad and not is_text_p(n) and not is_image_p(n)]
        rest_params_rest = [p for n, p in named_parameters if include(n, p) and p.requires_grad and not is_text_p(n) and not is_image_p(n)]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params_image, "weight_decay": 0.},
                {"params": rest_params_image, "weight_decay": args.wd},
                {"params": gain_or_bias_params_text, "weight_decay": 0.},
                {"params": rest_params_text, "weight_decay": args.wd},
                {"params": gain_or_bias_params_rest, "weight_decay": 0.},
                {"params": rest_params_rest, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        if args.precision == 'amp':
            scaler = GradScaler()

    start_epoch = 0 
    train_data = get_train_data(args, (preprocess_train, preprocess_val), tokenizer=tokenizer, epoch=start_epoch)
    val_data = get_val_data(args, (preprocess_train, preprocess_val), dataset_mode=args.dataset_mode, tokenizer=tokenizer, epoch=start_epoch)
    assert len(train_data), 'At least one train or eval dataset must be specified.'
    assert len(val_data), 'At least one train or eval dataset must be specified.'

    scheduler = None
    if optimizer is not None:
        total_steps = train_data["train_croco"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    if args.evaluate_trained_models: #TODO: remove this code and add evaluate trained models here
        if args.model_path == "frozen_clip":
            print("clip clip clip")
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="open_clip", device=device)
        elif args.model_path == "frozen_negClip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="negClip", device=device)
        elif args.model_path == "frozen_Clip_ViT-L-14":
            print("model_path", args.model_path)
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="Clip_ViT-L-14", device=device)
        elif args.model_path == "frozen_ConvNext":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="ConvNext", device=device)
        elif args.model_path == "frozen_SigLip":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="SigLip", device=device)
        elif args.model_path == "frozen_COCA":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="COCA", device=device)
        elif args.model_path == "frozen_DFN":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="DFN", device=device)
        elif args.model_path == "frozen_EVA":
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="EVA", device=device)
        else:
            model_path = f"/users/mali37/scratch/models/v4/L14/{args.model_path}"
            trained_model, preprocess_train, preprocess_val, tokenizer = get_model(args, model_name="ViT-L-14", from_path=model_path, device=device)

        evaluate_data = get_val_data(args, (preprocess_train, preprocess_val), dataset_mode=args.dataset_mode, tokenizer=tokenizer, epoch=start_epoch)
        evaluate(trained_model, evaluate_data, start_epoch, args, device=device)
    else:
        global_step_counter = 0
        for epoch in range(start_epoch, args.epochs):
            print(f'Start epoch {epoch}')
            #TODO: ERROR WITH PRINT LINE (goes upto 131%)
            global_step_counter = train_one_epoch(model, train_data, epoch, optimizer, scaler, scheduler, args, global_step_counter, device=device)
            completed_epoch = epoch + 1

            # validation_datasets = ["croco", "random", "croco_d", "stative", "vg_relation"]
            validation_datasets = ["vg_relation"]
            for dataset_name in validation_datasets:
                val_data = get_val_data(
                    args=args,
                    preprocess_fns=(preprocess_train, preprocess_val),
                    dataset_mode=dataset_name,
                    tokenizer=tokenizer,
                    epoch=completed_epoch
                )
                evaluate(model, val_data, dataset_name, completed_epoch, args, device=device)

if __name__ == "__main__":
    main()