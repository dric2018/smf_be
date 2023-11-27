"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 27 Nov, 2023
"""

import config

import lightning.pytorch as pl

from matplotlib import pyplot

import numpy as np

import os

from tqdm import tqdm
import timm
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoConfig

from transformer import generate_causal_attention_mask

import wandb


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
    wandb_logging:bool=False
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
                fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                fig.savefig(f"{fn}.png")                 
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
                fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                fig.savefig(f"{fn}.png")                 
            
                pyplot.close()


def greedy_decoding(
    model:pl.LightningModule, 
    batch_inp:dict, 
    max_len:int=config.MAX_OUT_SEQ_LEN, 
    debug:bool=False
):
    if model.device.type == "cpu":
        model.to(config.DEVICE)
    model.eval()
    
    sos_token = config.TARGETS_MAPPING["[SOS]"]
    eos_token = config.TARGETS_MAPPING["[EOS]"]
    
    input_ids=batch_inp["ids"].to(config.DEVICE)
    attn_mask=batch_inp["mask"].to(config.DEVICE)
    token_type_ids=batch_inp["token_type_ids"].to(config.DEVICE)
    imgs=batch_inp["in_state"].to(config.DEVICE)
    src_mask=(
        batch_inp["source_mask"].to(config.DEVICE), 
        batch_inp["source_mask_tokens"].to(config.DEVICE)
    )

    text_enc_last_h, learned_tokens = model._encode(
        input_ids=input_ids, 
        attn_mask=attn_mask, 
        token_type_ids=token_type_ids, 
        imgs=imgs    
    )
    
    decoder_inp = torch.empty(1, 1, dtype=torch.long, device=input_ids.device).fill_(sos_token)

    # decoding procedure
    for t in range(max_len):
        
        decoder_mask = generate_causal_attention_mask(
            dim=decoder_inp.shape[1]
        ).type_as(attn_mask)
        
        # generate predictions
        with torch.no_grad():
            logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = model._decode(
            decoder_inp=decoder_inp, 
            encoder_outs=(text_enc_last_h, learned_tokens), 
            src_mask=src_mask, 
            target_mask=decoder_mask,
            debug=debug,
            return_actions=False
        )

        # perform greedy decoding
        probs = model.decoder.action_generator(logits[:, -1])
            
        _, next_tok = torch.max(probs, dim=-1)
        # update decoder input
        decoder_inp = torch.cat((decoder_inp, next_tok.unsqueeze(1)), dim=1)
            
    return decoder_inp[:, 1:].cpu().detach(), logits, self_attn_ws.cpu().detach(), cross_attn_ws_seq.cpu().detach(), cross_attn_ws_tokens.cpu().detach()

    
def decode_predictions(predicted_ids:torch.Tensor)->list:
    
    curr_preds = []
    
    for tok in predicted_ids.tolist():
        if tok == 2:
            # EOS token encountered
            break
        else:
            curr_preds.append(config.TARGETS_REVERSE_MAPPING[tok])
    
    return " ".join(curr_preds)

def fetch_sample_from_batch(
    batch, 
    batch_size:int, 
    random:bool=False
):
    
    if random:
        idx = np.random.randint(batch_size)
    else:
        idx = 0
    
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

# class TelegramCallback(Callback):
#     """
#         Courtesy of Robert Bracco (Made-Up Masters@medium.com)

#         source: https://medium.com/@robertbracco1/how-to-write-a-telegram-bot-to-send-messages-with-python-part-2-1d9bf6ddc652
#     """
    
#     def on_epoch_end(self, trainer, pl_module):
#         include = ["epoch", "train_loss", "val_loss"]
#         d = {k:round(float(v), 3) for k,v in         
#              trainer.callback_metrics.items() if k in include}
        
#         messages = [f"{k} : {v}" for k,v in d.items()]
        
#         try:
#             telegram_send.send(messages=[messages])
#         except Exception as e: 
#             print("Unable to send:", e)


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
    src_mask=(batch["source_mask"].to(config.DEVICE), batch["source_mask_tokens"].to(config.DEVICE))
    target_mask=batch["target_mask"].to(config.DEVICE)
    
    # forward
    logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = model(
        input_ids=input_ids, 
        attn_mask=attn_mask, 
        token_type_ids=token_type_ids, 
        imgs=imgs,
        decoder_inp=decoder_inp, 
        src_mask=src_mask, 
        target_mask=target_mask 
    )

    # loss computation
    labels = batch["motor_cmd"]["labels"].to(config.DEVICE)
    loss = loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1))
        
    return loss, logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens

def validation_step(batch, model, loss_fn, debug:bool=False):
    inp = fetch_sample_from_batch(
        batch, 
        batch_size=batch["in_state"].shape[0],
        random=True
    )
    
    pred_ids, logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = greedy_decoding(
        model=model, 
        batch_inp=inp, 
        debug=debug
    )
    
    labels = inp["labels"].to(config.DEVICE)
    
    preds = model.decode_predictions(
            predicted_ids=pred_ids
    )[0]

    label = model.decode_predictions(
        predicted_ids=labels
    )[0]  
    
    # compute metrics
    val_loss = loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1)).item()  # loss
    cer = model.cer_fn(preds, label).item() # Character Error Rate
    wer = model.wer_fn(preds, label).item() # Word Error Rate
    
    output = {
        "val_loss"              : val_loss,
        "CER"                   : cer,
        "WER"                   : wer,
        "label"                 : label,
        "pred_ids"              : pred_ids,
        "pred_tokens"           : preds,
        "self_attn_ws"          : self_attn_ws, 
        "cross_attn_ws_seq"     : cross_attn_ws_seq, 
        "cross_attn_ws_tokens"  : cross_attn_ws_tokens
    }
    
    return output

def run_experiment(model, dm, opt, loss_fn, scheduler):
    
    loss_epoch = np.inf
    val_loss = np.inf
    best_val_loss = np.inf
    
    cer_ = np.inf
    wer_ = np.inf
        
    for e in range(config.EPOCHS):        
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
            loss, logits, self_attn_ws, cross_attn_ws_seq, _ = training_step(
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
                wandb_logging=True
            )

            plot_attention(
                cross_attn_ws_seq,
                kind="cross", 
                pre_fix="train_crossattn", 
                show=False, 
                folder="train",
                epoch=e,
                wandb_logging=True
            )   
            
            running_loss += loss.item()         
            
            # logging
            if step % 10 == 0:
                pbar.set_postfix(
                    train_loss_step="{:.04f}".format(running_loss/(step+1)),
                    train_loss="{:.04f}".format(loss_epoch),
                    CER="{:.04f}".format(cer_),
                    WER="{:.04f}".format(wer_),
                    val_loss="{:.04f}".format(val_loss),
                )
                pbar.update()

            # backward
            loss.backward()
            
            # Adjust learning weights
            opt.step()
            
        loss_epoch = running_loss / len(dm.train_dataloader())   
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
            
            cer_ = model.cer_fn(pred, label).item()
            wer_ = model.wer_fn(pred, label).item()
            f.write(f"Predicted \t: {pred}\n")
            f.write(f"Actual \t\t: {label}\n")
            
        # validation
        batch = next(iter(dm.val_dataloader()))
        out = validation_step(model=model, batch=batch, loss_fn=loss_fn)
        val_loss = out["val_loss"]
        
        # start scheduling lr after epoch X
        # X set to 30 to start us of
        if e >=config.LR_SCHEDULE_START:
            scheduler.step(val_loss)
       
        # plot attention weights
        plot_attention(
            out["self_attn_ws"], 
            show=False, 
            pre_fix="val_selfattn", 
            folder="val",
            epoch=e,
            wandb_logging=True
        )

        plot_attention(
            out["cross_attn_ws_seq"],
            kind="cross", 
            pre_fix="val_crossattn", 
            show=False, 
            folder="val",
            epoch=e,
            wandb_logging=True
        )   

        # plot_attention(
        #     out["cross_attn_ws_tokens"], 
        #     pre_fix="val_crossattn_tokens", 
        #     show=False, 
        #     folder="val",
        #     epoch=e,
        #     wandb_logging=True
        # )   
        
        # update best score
        if val_loss < best_val_loss:
            # save checkpoint
            path = os.path.join(config.MODEL_PATH, "be_model.bin")
            torch.save({
                'model_state_dict'      :model.state_dict(),
                'optimizer_state_dict'  :opt.state_dict(),
                'val_loss'              : val_loss, 
                'epoch'                 : e
                }, path)
            
            # update best score
            best_val_loss = val_loss        
        
        pbar.set_postfix(
            train_loss_step="{:.04f}".format(running_loss/(step+1)),
            train_loss="{:.04f}".format(loss_epoch),
            # CER="{:.04f}".format(cer_),
            # WER="{:.04f}".format(wer_),
            val_Loss="{:.04f}".format(best_val_loss),
            val_CER="{:.04f}".format(out["CER"]),
            val_WER="{:.04f}".format(out["WER"]),
            lr_epoch="{:.1e}".format(final_lr_epoch),
        )  
        pbar.update()
        
        logs_dict = {
            "epoch" :e,
            "train_loss":loss_epoch,
            "val_loss":val_loss,
            "val_CER":out["CER"],
            "valWER":out["WER"],
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
            f.write(f"Curr val loss \t\t: {val_loss:.5f}\n") 
            f.write(f"Best loss: \t\t: {best_val_loss:.5f}\n\n") 

        pbar.close()
        torch.cuda.empty_cache()
        
    return model

