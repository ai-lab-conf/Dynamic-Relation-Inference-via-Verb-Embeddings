import argparse
import os
import sys

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-images-for-croco",
        type=str,
        default="~/data/rand_simplerel_actionable_image_splits/train2017", #d1
        # default = "~/data/ds2/CROCO_D_Images_Final/train2017", #ds2
        # default=None,
        # default="~/data/new_splits/train2017", #og splits
        help="train image directory",
    )
    parser.add_argument(
        "--train-captions-for-croco",
        type=str,
        default="~/data/final_wc_data/index_to_captions_rand_simplerel_actionable_train.json", #ds1
        # default="~/data/ds2/ds2_index_to_captions_train.json", #ds2

        # default=None,
        # default="~/new_splits/index_to_caption_train.json", #og splits
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--train-images-for-croco-d",
        type=str,
        default = "~/data/ds2/CROCO_D_Images_Final/train2017", #ds2
        # default=None,
        # default="~/data/new_splits/train2017", #og splits
        help="train image directory",
    )
    parser.add_argument(
        "--train-captions-for-croco-d",
        type=str,
        default="~/data/ds2/ds2_index_to_captions_train.json", #ds2

        # default=None,
        # default="~/new_splits/index_to_caption_train.json", #og splits
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--train-images-for-stative",
        type=str,
        default="~/data/rand_simplerel_stative_image_splits/train2017",#ds3
        # default=None,
        # default="~/data/new_splits/train2017", #og splits
        help="train image directory",
    )
    parser.add_argument(
        "--train-captions-for-stative",
        type=str,
        default="~/data/final_wc_data/index_to_captions_rand_simplerel_stative_train.json", #ds3

        # default=None,
        # default="~/new_splits/index_to_caption_train.json", #og splits
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--val-images-for-croco",
        type=str,
        default="~/data/rand_simplerel_actionable_image_splits/test2017",#ds1
        help="val image directory",
    )
    parser.add_argument(
        "--val-captions-for-croco",
        type=str,
        default="~/data/final_wc_data/index_to_captions_rand_simplerel_actionable_test.json", #ds1
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--val-images-for-croco-d",
        type=str,
        default = "~/data/ds2/CROCO_D_Images_Final/test2017", #ds2
        help="val image directory",
    )
    parser.add_argument(
        "--val-captions-for-croco-d",
        type=str,
        default="~/data/ds2/ds2_index_to_captions_test.json", #ds2
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--val-images-for-stative",
        type=str,
        default="~/data/rand_simplerel_stative_image_splits/test2017",#ds3
        help="val image directory",
    )
    parser.add_argument(
        "--val-captions-for-stative",
        type=str,
        default="~/data/final_wc_data/index_to_captions_rand_simplerel_stative_test.json", #ds3
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
    parser.add_argument(
        "--val-images-for-vg-relation",
        type=str,
        default="~/data/final_vg_relations_images",#ds4
        help="val image directory",
    )
    parser.add_argument(
        "--val-captions-for-vg-relation",
        type=str,
        default="~/data/final_wc_data/vg_relation_index_to_captions.json", #ds4
        help="for WC json input, filepath to a list of caption indices with at least 2 WC options",
    )
  
   
    parser.add_argument(
        "--num-wc",
        type=str,
        default=1,
        help="number of weakly caption to use",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--delta-i", type=int, default=1.2233, help=" "
    )
    parser.add_argument(
        "--delta-t", type=int, default=0.615, help=" "
    )
    parser.add_argument(
        "--evaluate-trained-models", type=bool, default=False, help=" "
    )
    parser.add_argument(
        "--wandb-sweep", type=bool, default=True, help=" "
    )
    parser.add_argument(
        "--debug-master", type=bool, default=True, help=" "
    )
    parser.add_argument(
        "--loss", type=int, default=3, help="1 = clipLoss, 2 = hard negative loss, 3 = runLoss1, 4 = runLoss2"
    )

    parser.add_argument("--lr", type=float, default=1.8748186634898305e-05, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=4.5269353639276774e-05, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default= 1430, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--dataset-mode", type=str, default="croco_d", choices=["croco", "random", "croco_d", "stative", "vg_relation"],
        help="Options are [croco, random, croco_d, stative, vg_relation] (default: wc)"
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="open_clip",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="path of the trained model you want to evaluate",
    )
    parser.add_argument(
        "--pretrained",
        default='openai',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=True,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default='env://',
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='wandb',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=1.0, help="Gradient clip."
    )
    args = parser.parse_args()
    if args.train_images_for_croco:
        args.train_images_for_croco = os.path.expanduser(args.train_images_for_croco)
    if args.train_captions_for_croco:
        args.train_captions_for_croco = os.path.expanduser(args.train_captions_for_croco)
    if args.train_images_for_croco_d:
        args.train_images_for_croco_d = os.path.expanduser(args.train_images_for_croco_d)
    if args.train_captions_for_croco_d:
        args.train_captions_for_croco_d = os.path.expanduser(args.train_captions_for_croco_d)
    if args.train_images_for_stative:
        args.train_images_for_stative = os.path.expanduser(args.train_images_for_stative)
    if args.train_captions_for_stative:
        args.train_captions_for_stative = os.path.expanduser(args.train_captions_for_stative)
    
    if args.val_images_for_croco: 
        args.val_images_for_croco = os.path.expanduser(args.val_images_for_croco)
    if args.val_captions_for_croco:
        args.val_captions_for_croco = os.path.expanduser(args.val_captions_for_croco)
    
    if args.val_images_for_croco_d: 
        args.val_images_for_croco_d = os.path.expanduser(args.val_images_for_croco_d)
    if args.val_captions_for_croco_d:
        args.val_captions_for_croco_d = os.path.expanduser(args.val_captions_for_croco_d)

    if args.val_images_for_stative: 
        args.val_images_for_stative = os.path.expanduser(args.val_images_for_stative)
    if args.val_captions_for_stative:
        args.val_captions_for_stative = os.path.expanduser(args.val_captions_for_stative)

    if args.val_images_for_vg_relation: 
        args.val_images_for_vg_relation = os.path.expanduser(args.val_images_for_vg_relation)
    if args.val_captions_for_vg_relation:
        args.val_captions_for_vg_relation = os.path.expanduser(args.val_captions_for_vg_relation)
        
    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
    