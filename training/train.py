import json
import math
import os
import time
import torch.nn as nn
from contextlib import suppress
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import zip_longest
from open_clip import ClipLoss

def wcsim_loss(anchor_embeddings, wc_embeddings, device):
    """
    Computes weakly contrastive similarity loss for each anchor image with its weakly contrastive set.
    
    Args:
    - anchor_embeddings: tensor of anchor embeddings of shape (B, D).
    - wc_embeddings: tensor of weakly contrastive embeddings of shape (B, K-1, D).
    - gamma: temperature parameter for softmax.
    
    Returns:
    - loss: mean of WCSimLoss across all samples in the batch.
    """
    similarities = torch.stack([
        (F.cosine_similarity(anchor_embeddings, wc, dim=-1) + 1) * 0.5 for wc in wc_embeddings
    ], dim=-1).to(device) # Shape: (B, K-1)

    labels = torch.zeros(1, similarities.size(dim=1)).to(device)
    loss = F.binary_cross_entropy_with_logits(similarities, labels)
    
    return loss

def run_loss(clip_loss, wcsim_loss, delta):
    return (clip_loss * (1 + wcsim_loss * delta)) / 2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, global_step_counter, device):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model = unwrap_model(model)

    model.train()
    clip_loss = ClipLoss()

    dataloader = data['train_croco'].dataloader
    dataloader_rnd = data['train_random'].dataloader
    dataloader_croco_d = data['train_croco_d'].dataloader

    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for batch_num, (batch, rnd_batch, croco_d_batch) in enumerate(zip_longest(dataloader, dataloader_rnd, dataloader_croco_d)):
        if batch is not None:
            train_one_batch(global_step_counter, batch, loss_m, batch_time_m, data_time_m, end, 
                          optimizer, model, args, clip_loss, scaler, num_batches_per_epoch, 
                          sample_digits, autocast, epoch, scheduler, batch_num, device)
            global_step_counter += 1
            
        if rnd_batch is not None:
            train_one_batch(global_step_counter, rnd_batch, loss_m, batch_time_m, data_time_m, end,
                          optimizer, model, args, clip_loss, scaler, num_batches_per_epoch,
                          sample_digits, autocast, epoch, scheduler, batch_num, device)
            global_step_counter += 1

        if croco_d_batch is not None:
            train_one_batch(global_step_counter, croco_d_batch, loss_m, batch_time_m, data_time_m, end,
                          optimizer, model, args, clip_loss, scaler, num_batches_per_epoch,
                          sample_digits, autocast, epoch, scheduler, batch_num, device)
            global_step_counter += 1

    return global_step_counter

def train_one_batch(step, batch, loss_m, batch_time_m, data_time_m, end, optimizer, model, 
                   args, clip_loss, scaler, num_batches_per_epoch, sample_digits, autocast, 
                   epoch, scheduler, batch_num, device):
    scheduler(step)

    images, texts = batch
    images = images.to(device)
    texts = texts.to(device)

    optimizer.zero_grad()
    with autocast():
        image_features, text_features, logit_scale = model(images, texts)
        
        batch_size = args.batch_size
        
        #cliploss
        if args.loss == 1:
            avg_loss = clip_loss(image_features, text_features, logit_scale)

        #runloss
        else:
            run_loss_image_list = []
            run_loss_text_list = []
            
            for i in range(batch_size):
                start_idx = i * (args.num_wc + 1)
                end_idx = start_idx + args.num_wc + 1
                item_image_features = image_features[start_idx:end_idx]
                item_text_features = text_features[start_idx:end_idx]

                anchor_image_features = item_image_features[0].unsqueeze(0)
                anchor_text_features = item_text_features[0].unsqueeze(0)
                wc_image_features = item_image_features[1:]
                wc_text_features = item_text_features[1:]

                logits = (logit_scale * item_image_features @ item_text_features.T)
                labels = torch.arange(logits.shape[0], device=logits.device)

                clip_loss_value = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    
                image_wcsim_loss = wcsim_loss(anchor_image_features, wc_image_features, device)
                text_wcsim_loss = wcsim_loss(anchor_text_features, wc_text_features, device)
                
                run_loss_image = run_loss(clip_loss_value, image_wcsim_loss, args.delta_i)
                run_loss_text = run_loss(clip_loss_value, text_wcsim_loss, args.delta_t)

                run_loss_image_list.append(run_loss_image)
                run_loss_text_list.append(run_loss_text)


            mean_run_loss_image = torch.sum(torch.stack(run_loss_image_list)).to(device)
            mean_run_loss_text = torch.sum(torch.stack(run_loss_text_list)).to(device)
            avg_loss = ((mean_run_loss_image + mean_run_loss_text) / 2)

    if scaler is not None:
        scaler.scale(avg_loss).backward()
        if args.norm_gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        avg_loss.backward()
        if args.norm_gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
        optimizer.step()

    with torch.no_grad():
        unwrap_model(model).logit_scale.clamp_(0, math.log(10))

    batch_time_m.update(time.time() - end)
    end = time.time()
    batch_count = batch_num + 1
    if batch_count % 10 == 0 or batch_count == num_batches_per_epoch:
        num_samples = batch_count * batch_size
        percent_complete = 100.0 * batch_count / num_batches_per_epoch

        loss_m.update(avg_loss.item(), batch_size)
        logit_scale_scalar = logit_scale.item()
        print(
            f"Train Epoch: {epoch} [{num_samples:>{sample_digits}} ({percent_complete:.0f}%)] "
            f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
            f"Data (t): {data_time_m.avg:.3f} "
            f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size / batch_time_m.val:#g}/s "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} "
            f"Logit Scale: {logit_scale_scalar:.3f}"
        )

        batch_time_m.reset()
        data_time_m.reset()

def evaluate(model, data, dataset_mode, epoch, args, device):
    print("Starting evaluation function")
    metrics = {}
    model = unwrap_model(model)
    model.eval()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if f'val_{dataset_mode}' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data[f'val_{dataset_mode}'].dataloader
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing retrieval scores"):
                image_options = []
                for i_option in batch["image_options"]:
                    image_embeddings = model.encode_image(i_option.to(device))
                    image_embeddings = image_embeddings.cpu().numpy()
                    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
                    image_options.append(np.expand_dims(image_embeddings, axis=1))

                caption_options = []
                for c_option in batch["caption_options"]:
                    c_option = c_option.squeeze()
                    caption_embeddings = model.encode_text(c_option.to(device))
                    caption_embeddings = caption_embeddings.cpu().numpy()
                    caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
                    caption_options.append(np.expand_dims(caption_embeddings, axis=1))

                image_options = np.concatenate(image_options, axis=1)
                caption_options = np.concatenate(caption_options, axis=1)
                batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)
                scores.append(batch_scores)
        
        all_scores = np.concatenate(scores, axis=0)
        result_records = evaluate_scores(all_scores, True)
        metrics.update({
            **result_records[0],
            "epoch": epoch,
        })

    if metrics:
        print(f"Evaluation Epoch: {epoch} " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))

    return metrics

def evaluate_scores(scores, anchor_only):
    """
    Parameters:
    - scores: N x NumImages x NumCaptions
    - anchor_only: whether to calculate the accuracy for t2i and i2t only with respect to the anchor prediction
    Returns: dictionary of accuracies for t2i and i2t
    """
    scores_i2t = scores
    num_images = scores_i2t.shape[1]
    scores_t2i = np.transpose(scores, axes=[0, 2, 1])
    num_captions = scores_t2i.shape[1]
    assert (num_images <= num_captions)
    scores_t2i = scores_t2i[:, :num_images, :]

    if not anchor_only:
        preds_per_image = np.argmax(scores_i2t, axis=-1)
        answer_per_image = np.tile(np.arange(num_images), (preds_per_image.shape[0], 1))
        preds_per_text = np.argmax(scores_t2i, axis=-1)
        answer_per_text = np.tile(np.arange(num_images), (preds_per_text.shape[0], 1))
    else:
        preds_per_image = np.argmax(scores_i2t, axis=-1)[:,:1]
        answer_per_image = np.tile(0, (preds_per_image.shape[0], 1))
        preds_per_text = np.argmax(scores_t2i, axis=-1)[:,:1]
        answer_per_text = np.tile(0, (preds_per_text.shape[0], 1))
            
    i2t_correct_mask = (preds_per_image == answer_per_image)
    i2t_accuracy = i2t_correct_mask.mean()

    t2i_correct_mask = (preds_per_text == answer_per_text)
    t2i_accuracy = t2i_correct_mask.mean()

    return [{"image_to_text accuracy": i2t_accuracy,
             "text_to_image accuracy": t2i_accuracy}]