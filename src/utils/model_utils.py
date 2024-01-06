"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 6 Jan, 2024
"""

import config

import lightning.pytorch as pl
import Levenshtein
import logging
logging.basicConfig(level="INFO")

from matplotlib import pyplot

import numpy as np

import os

import pandas as pd
import rt1
import timm

from tqdm.auto import tqdm 

import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformer import make_attn_mask

import wandb

def get_seq_len(x:torch.Tensor):
    non_padding_mask = (outp[0] != config.TGT_PAD_TOK_ID)
    seq_len = non_padding_mask.sum().item()
    return seq_len

class TextEncoder(nn.Module):
    def __init__(
        self, 
        dropout_rate:float=config.ENCODER_DROPOUT_RATE, 
        freeze:bool=True
    ):
        super().__init__()
        
        self.freeze = freeze
        
        model_config = AutoConfig.from_pretrained(config.LANG_MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(config.LANG_MODEL_NAME, config=model_config)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if self.freeze:
            self._freeze_model()

    def _freeze_model(self):
        for param in self.encoder.parameters():
            param.requires_grad = False 
        
    def forward(self, inp_ids, mask, tok_type_ids):
        # embed NL instructions
        text_enc = self.encoder(
            input_ids=inp_ids,
            attention_mask=mask,
            token_type_ids=tok_type_ids
        )
        
        
        # print(text_enc.shape)
        out = self.dropout(text_enc.pooler_output)
        
        return out, text_enc.last_hidden_state
    
    
def get_backbone(model:nn.Module):
    return nn.Sequential(*list(model.children())[:-2])

def conv(ic, oc, k, s, p, activation:str="GELU"):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """    
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        getattr(nn, activation)(),
        nn.BatchNorm2d(oc),
    )


class ImageFeatureExtractor(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(
        self, 
        pretrained:bool=True, 
        arch:str="efficientnet_b3",
        freeze:bool=True
    ):
        super().__init__()

        self.pretrained = pretrained
        self.freeze = freeze

        self.fe     = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
            features_only=True,
        )
            
        self.out_layer = VisionLanguageHead(prev_channels=config.NUM_CHANNELS[arch], n_classes=config.EMBEDDING_DIM)

            
        if self.freeze:
            self._freeze_model()
            
    def _freeze_model(self):
        for param in self.fe.parameters():
            param.requires_grad = False 

    def forward(self, x, flat_out:bool=False, return_feats:bool=True):
        
        # extract image features
        enc = self.fe(x)[-1] # return last output features
        out = self.out_layer(enc, return_feats)
        
        if flat_out:
            return torch.flatten(out, 1)
        else:
            return out

class Head(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(self, prev_channels, n_classes):
        super(Head, self).__init__()

        self.conv = nn.Conv2d(prev_channels, n_classes, 1, 1, 0)
        self.activation = nn.GELU()
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.fc = nn.Sequential(nn.Linear(512, 1024),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(1024, 1024),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(1024, n_classes))

    def forward(self, x, return_feats:bool=True):
        
        print(f"Head in: {x.shape}")
        feats = self.conv(x)
        feats = self.global_max_pool(feats)

        if return_feats:
            return feats
        else:
            x = feats.view(feats.size(0), feats.size(1))
            x = self.fc(x)
            return x

class VisionLanguageHead(nn.Module):
    def __init__(self, prev_channels, n_classes):
        super(VisionLanguageHead, self).__init__()

        self.conv = nn.Conv2d(prev_channels, n_classes, 1, 1, 0)
        self.activation = nn.GELU()
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, return_feats:bool=True):
        
        # print(f"Head in: {x.shape}")
        feats = self.activation(self.conv(x))
        # feats = 

        if return_feats:
            return feats
        else:
            return self.global_max_pool(feats)

def plot_attention(
    attn_w,
    pre_fix:str="train",
    example_idx:int=0, 
    kind:str=None, 
    show:bool=True,
    epoch:int=0,
    folder:str="train",
    wandb_logging:bool=config.WANDB_LOGGING
):
    
    num_layers = 1
    
    if attn_w.ndim > 3:
        if attn_w.ndim >4:
            # multi layer plot
            B, num_layers, num_heads, L1, L2 = attn_w.shape
        else:
            B, num_heads, L1, L2 = attn_w.shape
        
        # self-attention plot by default
        x_axis_title = "Input seq"
        y_axis_title = "Input seq"          
        
        if (L1 != L2) or (kind=="cross"):
            # corss-attention plot
            x_axis_title = "Input"
            y_axis_title = "Output"                  
        
        fig, axes = pyplot.subplots(
            num_layers, 
            num_heads, 
            figsize=(15, 5),
            dpi=300
        )
        
        if num_layers and num_layers > 1:
            # multi-layer
            for l in range(num_layers):
                for head_idx in range(num_heads):
                    # Extract attention weights for the chosen example and head
                    attn_w_example_head = attn_w[example_idx, l, head_idx].cpu().detach().numpy()

                    # Visualize the attention weights as a heatmap in the corresponding subplot
                    ax = axes[l, head_idx]
                    ax.imshow(attn_w_example_head, cmap='GnBu', interpolation='nearest')
                    ax.set_title(f'Layer {l} - Head {head_idx + 1}')
                    ax.set_xlabel(x_axis_title)
                    ax.set_ylabel(y_axis_title)                
        else:
            # single-layer
            for head_idx in range(num_heads):
                # Extract attention weights for the chosen example and head
                attn_w_example_head = attn_w[example_idx, head_idx].cpu().detach().numpy()

                # Visualize the attention weights as a heatmap in the corresponding subplot
                ax = axes[head_idx]
                ax.imshow(attn_w_example_head, cmap='GnBu', interpolation='nearest')
                ax.set_title(f'Head {head_idx + 1}')
                ax.set_xlabel(x_axis_title)
                ax.set_ylabel(y_axis_title)        
        
        # suptitle
        if L1 == L2 and kind !="cross":
            pyplot.suptitle("Self-attention weights")
        else:
            pyplot.suptitle("Cross-attention weights")
            
        pyplot.tight_layout()

        if show:
            pyplot.show()
        else:              
            if wandb_logging:
                wandb.log({f"{pre_fix}_epoch_{epoch}": wandb.Image(fig)})    
                pyplot.close()
            else:
                # fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                # fig.savefig(f"{fn}.png")                 
                pyplot.close()
                        
    else:
        # Extract attention weights for the chosen example
        attn_w_example = attn_w[example_idx].cpu().detach().numpy()

        # Visualize the attention weights as a heatmap in the corresponding subplot
        pyplot.imshow(attn_w_example, cmap='GnBu')
        pyplot.title(f'Attention weights')
    
        pyplot.tight_layout()

        if show:
            pyplot.show()
        else:            
            if wandb_logging:
                wandb.log({f"{pre_fix}_epoch_{epoch}": wandb.Image(fig)})    
                pyplot.close()
            else:
                # fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                # fig.savefig(f"{fn}.png")                 
            
                pyplot.close()

@torch.inference_mode()
def greedy_decoding(
    model:pl.LightningModule=None, 
    batch_inp:dict=None, 
    max_len:int=config.MAX_OUT_SEQ_LEN, 
    debug:bool=False,
    device:"str"=config.DEVICE
):
    if model.device.type == "cpu":
        model.to(device)
    model.eval()
    
    sos_token = config.SOS_TOKEN_ID
    eos_token = config.EOS_TOKEN_ID
    
    input_ids=batch_inp["ids"].to(device)
    attn_mask=batch_inp["mask"].to(device)
    token_type_ids=batch_inp["token_type_ids"].to(device)
    imgs=batch_inp["in_state"].to(device)

    _, learned_tokens = model._encode(
        input_ids=input_ids, 
        attn_mask=attn_mask, 
        token_type_ids=token_type_ids, 
        imgs=imgs    
    )
    
    decoder_inp = torch.empty(1, 1, dtype=torch.long, device=input_ids.device).fill_(sos_token)

    for t in range(config.MAX_OUT_SEQ_LEN):
        mask = make_attn_mask(dim=decoder_inp.shape[1])

        # with torch.no_grad():
        logits, self_attn_ws, cross_attn_ws = model._decode(
            decoder_inp=decoder_inp, 
            encoder_out=learned_tokens,
            attn_mask=mask,
            return_actions=False
        )

        # perform greedy decoding
        probs = model.decoder.action_generator(logits[:, -1])

        _, next_tok = torch.max(probs, dim=-1)
        
        # stop decoding if 2nd token is "EOS"
        if t>=1 and next_tok == eos_token:
            break
        # update decoder input
        decoder_inp = torch.cat((decoder_inp, next_tok.unsqueeze(1)), dim=1)
            
    return decoder_inp[:, 1:].cpu().detach(), logits, self_attn_ws.cpu().detach(), cross_attn_ws.cpu().detach()

    
def decode_predictions(predicted_ids:torch.Tensor)->list:
    
    curr_preds = []
    
    for tok in predicted_ids.tolist():
        if tok == config.EOS_TOKEN_ID:
            # EOS token encountered
            break
        else:
            curr_preds.append(config.TARGETS_REVERSE_MAPPING[tok])
    
    return " ".join(curr_preds)

def fetch_sample_from_batch(
    batch, 
    batch_size:int, 
    random:bool=False,
    idx:int=0
):
    
    if random:
        idx = np.random.randint(batch_size)
    else:
        assert idx < batch_size, f"Value of idx ({idx}) is higher than batch size ({batch_size})"
        idx = idx
    
    sample  = {}
    
    for k, v in batch.items():
        if k in ["action_desc", "motor_cmd"]:
            for k1 in v.keys():
                if k1 == "raw":
                    sample[k1+"_"+k] = v[k1][idx]
                else:
                    sample[k1] = v[k1][idx].unsqueeze(0)
        else:
            sample[k] = v[idx].unsqueeze(0)
                    
    return sample


def has_nan(x:torch.Tensor):
    return torch.isnan(x).any().item()


class StopTrainingException(Exception):
    pass


## Training utils
class CustomWandbTable:
    def __init__(self, columns):
        self.columns = columns
        self.data = {col: [] for col in columns}

    def update(self, values):
        for col, val in zip(self.columns, values):
            self.data[col].append(val)

    def log_to_wandb(self):
        wandb.log(self.data)
        wandb.run.log({"training_summary": self.data})

        
def training_step(model, batch, loss_fn):

    input_ids=batch["action_desc"]["ids"].to(config.DEVICE)
    attn_mask=batch["action_desc"]["mask"].to(config.DEVICE)
    token_type_ids=batch["action_desc"]["token_type_ids"].to(config.DEVICE)
    imgs=batch["in_state"].to(config.DEVICE)
    decoder_inp=batch["motor_cmd"]["decoder_inp_ids"].to(config.DEVICE)
    
    # forward
    logits, self_attn_ws, cross_attn_ws = model(
        input_ids=input_ids, 
        attn_mask=attn_mask, 
        token_type_ids=token_type_ids, 
        imgs=imgs,
        decoder_inp=decoder_inp
    )

    # loss computation
    labels = batch["motor_cmd"]["labels"].to(config.DEVICE)
    loss = loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1))
        
    return loss, logits, self_attn_ws, cross_attn_ws


def validation_step(batch, model, loss_fn, debug:bool=False):
    
    inp = fetch_sample_from_batch(
        batch, 
        batch_size=batch["in_state"].shape[0],
        random=True
    )
    
    pred_ids, logits, self_attn_ws, cross_attn_ws = greedy_decoding(
        model=model, 
        batch_inp=inp, 
        debug=debug
    )
    
    labels = inp["labels"].to(config.DEVICE)
    
    preds = model.decode_predictions(
            predicted_ids=pred_ids
    )
    
    # print(pred_ids)

    label = model.decode_predictions(
        predicted_ids=labels
    )
    
    lev_dist = calc_edit_distance(preds, labels, batch=True)
    
    # compute metrics
    # print(logits.shape, labels.shape)
    # val_loss = loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1)).item()  # loss
    # cer = model.cer_fn(preds[0], label[0]).item() # Character Error Rate
    # wer = model.wer_fn(preds[0], label[0]).item() # Word Error Rate
    
    output = {
        "label"                 : label[0],
        "pred_tokens"           : preds[0],
        "self_attn_ws"          : self_attn_ws, 
        "cross_attn_ws"         : cross_attn_ws,
        "logits":logits,
        "dist": lev_dist
    }
    
    return output

def load_checkpoint(model_name:str="be_model", device:str=config.DEVICE):
    
    logging.info("Loading model from checkpoint...")
    
    logging.info("Creating instance of RTCRAM...")
    
    model = rt1.RTCRAM(
        cnn_bacnbone=config.SELECTED_CNN_BACKBONE, 
        num_res_blocks=config.NUM_RES_BLOCKS,
        freeze_cnn_backbone=config.FREEZE_CNN,
        args=None
    ).to(device)
    
    logging.info("Preparing checkpoint...")
    CKPT_PATH = os.path.join(config.MODEL_PATH, model_name+".bin")
    ckpt = torch.load(CKPT_PATH)  
    
    logging.info("loading model state dict...")
    model.load_state_dict(ckpt["model_state_dict"])
    
    logging.info("Loading model from checkpoint...Complete!")
    
    return model

def run_experiment(
    model, 
    dm, 
    opt, 
    loss_fn, 
    scheduler, 
    resume_training:bool=False, 
    epoch_resume:int=0
):
    
    # setup data module
    dm.setup()
    
    if resume_training:
        # load checkpoint
        CKPT_PATH = os.path.join(config.MODEL_PATH, "be_model.bin")
        ckpt = torch.load(CKPT_PATH)
        EPOCH_RESUME = ckpt["epoch"] if ckpt["epoch"] > 0 else epoch_resume
        
        # load model state dict
        model = load_checkpoint()
        
        # load optimizer state dict
        opt.load_state_dict(ckpt["optimizer_state_dict"])

        # setup metrics reporting
        val_dist, best_val_dist = ckpt["val_dist"], ckpt["val_dist"]
        perplexity, best_perplexity = ckpt["perplexity"], ckpt["perplexity"]
        
    else:
        # setup metrics reporting
        val_dist = 1e9
        best_val_dist = 1e9
        perplexity = 1e9
        best_perplexity = 1e9
        
    
    epoch_bar = range(config.EPOCHS)
    if resume_training:
        epoch_bar = range(EPOCH_RESUME, config.EPOCHS)
    
    for e in epoch_bar:        
        running_loss = 0.
        num_steps = len(dm.train_dataloader())
        
        pbar = tqdm(
            range(num_steps),
            position=0,
            leave=True,
            dynamic_ncols=True,
            total = num_steps
        )
        
        # training
        model.train()
        for step, batch in enumerate(dm.train_dataloader()):   
            pct = 100. * step / num_steps
            pbar.set_description(
                f"Epoch {e+1}/{config.EPOCHS} - (Train {pct:.1f}%)"
            )
            pbar.update()
            
            opt.zero_grad()

            # training step
            loss, logits, self_attn_ws, cross_attn_ws = training_step(
                model=model, 
                batch=batch, 
                loss_fn=loss_fn
            )
            
            # plot attention weights
            plot_attention(
                self_attn_ws, 
                show=False, 
                pre_fix="train_selfattn", 
                folder="train",
                epoch=e,
                wandb_logging=config.WANDB_LOGGING
            )

            plot_attention(
                cross_attn_ws,
                kind="cross", 
                pre_fix="train_crossattn", 
                show=False, 
                folder="train",
                epoch=e,
                wandb_logging=config.WANDB_LOGGING
            )   
            
            running_loss += loss.item()
            train_loss = running_loss/(step+1)
            perplexity = torch.exp(loss)
            
            # logging
            if step % 10 == 0:
                pbar.set_postfix(
                    train_loss="{:.04f}".format(train_loss),
                    perplexity="{:.04f}".format(perplexity),
                    val_dist="{:.04f}".format(val_dist)
                )
                pbar.update()

            # backward
            loss.backward()
            
            # Adjust weights
            opt.step()
            
        final_lr_epoch = float(opt.param_groups[0]['lr'])
        
        # predictions
        preds = logits.softmax(dim=-1).argmax(dim=-1)

        # decode predictions
        preds = model.decode_predictions(
            predicted_ids=preds
        )

        labels = model.decode_predictions(
            predicted_ids=batch["motor_cmd"]["labels"]
        )         
            
        # log decoded sentenses
        with open(config.LOGGING_FILE, "a") as f:            
            f.write(f"Epoch #{e+1}\n")
            f.write(f"[Train] \n")
            
            pred = preds[0]
            label = labels[0]
            
            f.write(f"Predicted \t: {pred}\n")
            f.write(f"Actual \t\t: {label}\n")
            f.write(f"Loss \t\t: {train_loss:.3f}\n")
            f.write(f"Perplexity \t\t: {perplexity:.3f}\n")
                        
        # valid_pbar = tqdm(range(config.NUM_VAL_STEPS), desc="Validating...")
            
        for _ in range(config.NUM_VAL_STEPS):
            val_batch = next(iter(dm.val_dataloader()))
            out = validation_step(model=model, batch=val_batch, loss_fn=loss_fn)

            # Edit distance
            dist = out["dist"]
        
        # aggregate results
        val_dist = dist / config.NUM_VAL_STEPS

        # plot attention weights
        plot_attention(
            out["self_attn_ws"], 
            show=False, 
            pre_fix="val_selfattn", 
            folder="val",
            epoch=e,
            wandb_logging=config.WANDB_LOGGING
        )

        plot_attention(
            out["cross_attn_ws"],
            kind="cross", 
            pre_fix="val_crossattn", 
            show=False, 
            folder="val",
            epoch=e,
            wandb_logging=config.WANDB_LOGGING
        )   
        
        # update best score
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            
            if val_dist < best_val_dist:
                best_val_dist = val_dist
                        
            # save checkpoint
            path = os.path.join(config.MODEL_PATH, "be_model.bin")
            torch.save({
                'model_state_dict'      : model.state_dict(),
                'optimizer_state_dict'  : opt.state_dict(),
                'val_dist'              : best_val_dist,
                'perplexity': best_perplexity,
                'epoch'                 : e
                }, path)                
                
        
        pbar.set_postfix(
            train_loss="{:.04f}".format(train_loss),
            perplexity="{:.04f}".format(perplexity),
            val_dist="{:.04f}".format(val_dist),
            LR="{:.1e}".format(final_lr_epoch)
        )  
        pbar.update()
        
        logs_dict = {
            "epoch" :e,
            "train_loss":train_loss,
            "perplexity":perplexity,
            "Lev_dist": best_val_dist,
            "lr":final_lr_epoch
        }
        wandb.log(logs_dict)
        
        # log decoded sentenses
        with open(config.LOGGING_FILE, "a") as f:                        
            pred = out["pred_tokens"]
            label = out["label"]
            
            f.write(f"[Val] \n")            
            f.write(f"Predicted \t: {pred}\n")
            f.write(f"Actual \t\t: {label}\n") 
            f.write(f"Best Val dist \t\t: {best_val_dist:.3f}\n\n") 
            
        pbar.close()
        torch.cuda.empty_cache()
        
    return model


def lrfn(
    epoch:int, 
    lr_start:float=config.LR_START,
    lr_max:float = config.LR_MAX,
    lr_min:float = config.LR_MIN,
    num_epochs:int=config.EPOCHS,
    lr_exp_decay:float = config.LR_EXP_DECAY

):
    """
        Courtesy of Chris Deotte KGMON@NVIDIA
    """
    lr_rampup_epochs = num_epochs // (5*3)
    lr_sustain_epochs = num_epochs // (5*5)

    
    if epoch < lr_rampup_epochs:
        lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
    elif epoch < lr_rampup_epochs + lr_sustain_epochs:
        lr = lr_max
    else:
        lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
    return lr


def calc_edit_distance(predictions:list, y:list, batch:bool=False):

    dist                = 0
    
    if batch:
        batch_size = len(predictions)
        for idx in range(batch_size): 
            y_    = y[idx]
            pred_ = predictions[idx]
            # distance
            dist      += Levenshtein.distance(y_, pred_)

        dist    /= batch_size
    else:
        dist      += Levenshtein.distance(y, predictions)

    
    return dist


def inference_step(
    test_loader, 
    debug:bool=False, 
    mode:str="inference",
    device:str=config.TEST_DEVICE
):
    
    """
        Execute inference procedure in 2 modes
        
        mode: (str) One of 
            inference: generate predictions using trained model
            eval: Evaluate model by comparing predictions to ground truth
            
        test_loader: (DataLoader) Data loader to be used for testing
    """
    
    model = load_checkpoint(model_name="RTCRAM")
    
    output = {
        "self_attn_ws"          : [], 
        "cross_attn_ws"         : [],
        "preds"         : [],
        "labels" : []
    }
    logging.info("Running inference now...")

    
    test_progress = tqdm(range(len(test_loader)), desc="Running inference")
    
    for batch_num in test_progress:
        batch = next(iter(test_loader))
        
        for sample_id in tqdm(range(config.TEST_BATCH_SIZE), leave=False, desc="Generating motor commands"):
        
            inp = fetch_sample_from_batch(
                batch, 
                batch_size=batch["in_state"].shape[0],
                random=False,
                idx=sample_id
            )

            pred_ids, logits, self_attn_ws, cross_attn_ws = greedy_decoding(
                model=model, 
                batch_inp=inp, 
                debug=debug,
                device=device
            )


            preds = model.decode_predictions(
                    predicted_ids=pred_ids
            )

            output["self_attn_ws"].append(self_attn_ws)
            output["cross_attn_ws"].append(cross_attn_ws)
            output["preds"].append(preds[0])

            if mode == "eval":
                labels = inp["labels"].to(config.DEVICE)
                label = model.decode_predictions(
                    predicted_ids=labels
                )
            output["labels"].append(label[0])
        
        # break
        
    if mode == "eval":
        test_dist = calc_edit_distance(
            predictions=output["preds"], 
            y=output["labels"], 
            batch=True
        )        
        
        inference_results = pd.DataFrame({
            "prediction": output["preds"],
            "label": output["labels"],
            "correct": [float(p==l) for p,l in zip(output["preds"], output["labels"])],
            "distance": [calc_edit_distance(p, l, batch=False) for p,l in zip(output["preds"], output["labels"])]
        })
        
        success_rate = 100*inference_results.correct.mean()
        
        print(f"**** Evaluatiion Report *****")
        print(f"> Test Lev. distance\t: {test_dist:.4f}")
        print(f"> Success Rate\t\t: {success_rate:.4f}%")
        print(f"**** Evaluatiion Report *****")
        return inference_results
    else:
        return output