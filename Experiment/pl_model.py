
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchmetrics
from transformers import AutoModelForTokenClassification, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42, workers=True)

class LightninggarNER(pl.LightningModule):

    def __init__(self, model_name: str, num_labels: int, lr: float=1e-04, warmup_steps: int=100):
        super().__init__()
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(task='multiclass')
        self.val_acc = torchmetrics.Accuracy(task='multiclass')
        self.train_f1 = torchmetrics.F1(task='multiclass')
        self.val_f1 = torchmetrics.F1(task='multiclass')
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        #self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=number_of_labels) 
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config) 
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs
    
    #TODO: do I need to reset after end of epoch each metric?
    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs.loss
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        #optimizer = AdamW(self.parameters(), lr=1e-04)
        #num_tr_optim_steps = self.hparams.num_train_epochs * size_train
        num_tr_optim_steps = self.hparams.num_train_epochs * (total_size_train / (batch_size * grad_acc_steps)) 
        scheduler = get_linear_schedule_with_warmup(num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_tr_optim_steps)



# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=1000,    
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)

fast_garner = LightninggarNER("distilbert-base-uncased", 68, None, None)

ckp_path = ''
pl.Trainer(default_root_dir=ckp_path, enable_checkpointing=True, accumulate_grad_batches=2, accelerator="auto", devices="auto")
#class CRF(nn.Module):
#    def __init__(self, model_name, id_to_tag, output_size, device ='cpu', dropout_rate = 0.1):
#        super(CRF, self).__init__()
#        
#        self.id_to_tag = id_to_tag
#            
#        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True)
#
#        
#        # output layer
#        self.feedforward = nn.Linear(in_features= self.encoder.config.hidden_size, out_features= output_size)
#
#        self.crf_layer = ConditionalRandomField(num_tags= output_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
#
#        self.dropout = nn.Dropout(dropout_rate)
#
#        self.device = device
#
#    def update_mask(self, labels, attention_mask):
#
#        shape = labels.size()
#        labelsU = torch.flatten(labels)
#        attention_maskU = torch.flatten(attention_mask)
#        idx = (labelsU == -100).nonzero(as_tuple=False)
#        idx = torch.flatten(idx)
#        labelsU[idx] = torch.tensor(3)
#        attention_maskU[idx] = torch.tensor(0)
#        labelsU = labelsU.reshape(shape)
#        attention_maskU = attention_maskU.reshape(shape)
#
#        return labelsU, attention_maskU
#    
#    def forward(self, input_ids, attention_mask, labels):
#
#        batch_size = input_ids.size(0)
#        
#        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#        embedded_text_input = embedded_text_input.last_hidden_state
#        
#        #for baseline
#        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))
#
#        token_scores = self.feedforward(embedded_text_input)
#        
#        token_scores = F.log_softmax(token_scores, dim=-1)
#        
#
#        labelsU, attention_maskU = self.update_mask(copy.deepcopy(labels), copy.deepcopy(attention_mask))
#        
#        loss = -self.crf_layer(token_scores, labelsU, attention_maskU) / float(batch_size)
#        best_path = self.crf_layer.viterbi_tags(token_scores, attention_maskU)
#
#        preds = torch.full(labels.size(),-100) 
#
#        for i in range(batch_size):
#            idx,_ = best_path[i]
#            preds[i][:len(idx)] = torch.tensor(idx)
#
#        # print(preds.size())
#        # print(labels.size())
#        
#        return {'loss':loss, 'logits':preds.to(self.device)}

    
