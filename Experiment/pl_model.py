
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForTokenClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from datasets import Dataset

import sys
import os
sys.path.append('../')
from util.utils import get_tag_mappings, get_data, compute_metrics, compute_metrics_crf
from util.dataloader import PreDataCollator

SEED = 42
pl.seed_everything(SEED, workers=True)

class LightninggarNER(pl.LightningModule):

    def __init__(self, model_name: str, 
                 num_labels: int, total_size_train: int, num_train_epochs: int=7,
                 batch_size: int=8, lr: float=1e-04, warmup_steps: int=100):
        super().__init__()
        self.total_size_train = total_size_train
        self.save_hyperparameters()
        #TODO: should comment the following 4 lines and replace with compute_metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_labels)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_labels)
        self.train_f1 = MulticlassF1Score(num_classes=num_labels)
        self.val_f1 = MulticlassF1Score(num_classes=num_labels)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        #self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=number_of_labels) 
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config) 

        self.data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs
    
    #TODO: do I need to reset after end of epoch each metric?, and add logger argumnent to log
    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        #self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('train_f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        #self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('val_f1', self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    
    #def train_dataloader(self):
    #    return DataLoader(self.train_data, collate_fn=self.data_collator)

    #def val_dataloader(self):
    #    return DataLoader(self.dev_data, collate_fn=self.data_collator)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        #optimizer = AdamW(self.parameters(), lr=1e-04)
        #num_tr_optim_steps = self.hparams.num_train_epochs * size_train
        grad_acc_steps = 2  # this is ugly I'm sorry
        num_tr_optim_steps = self.hparams.num_train_epochs * (total_size_train / (self.hparams.batch_size * grad_acc_steps)) 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_tr_optim_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    
LANG = 'es'
MAX_LEN = 256
df = get_data(LANG, None, train=True)
        
train_df, dev_df = train_test_split(df, test_size=0.2, random_state=SEED)
if LANG!=None:
    train_df = train_df[train_df['lang']==LANG]
    dev_df = dev_df[dev_df['lang']==LANG]

train_df = train_df.iloc[:500]
dev_df = dev_df.iloc[:500]
total_size_train = train_df.shape[0]

train_data = Dataset.from_pandas(train_df)
dev_data = Dataset.from_pandas(dev_df)
tags_to_ids, ids_to_tags = get_tag_mappings()
num_of_labels = len(tags_to_ids)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids)
train_tokenized = train_data.map(collator, remove_columns=train_data.column_names, batch_size=4, num_proc=4, batched=True)
dev_tokenized = dev_data.map(collator, remove_columns=dev_data.column_names, batch_size=4, num_proc=4, batched=True)


data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')
train_dataloader = DataLoader(train_tokenized, collate_fn=data_collator)
val_dataloader = DataLoader(dev_tokenized, collate_fn=data_collator)
# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=1000,    
    save_top_k=10,
    monitor="val_loss",  # must share name with log str
    mode="min",
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)

max_epochs = 7
fast_garner = LightninggarNER("distilbert-base-uncased", num_train_epochs=max_epochs,
                              num_labels=num_of_labels, total_size_train=total_size_train)


ckp_path = './lightning_model'
trainer = pl.Trainer(
           max_epochs=max_epochs,
           min_epochs=1,
           default_root_dir=ckp_path,
           enable_checkpointing=True,
           accumulate_grad_batches=2,
           accelerator="auto",
           devices="auto")

trainer.fit(fast_garner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
raise SystemExit
trainer.test(fast_garner)  # reload from ckpt?