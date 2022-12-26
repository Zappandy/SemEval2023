
import torch.nn as nn
import torch.nn.functional as F
import torch
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from transformers import AutoModel
import copy


class CRF(nn.Module):
    def __init__(self, model_name, id_to_tag, output_size, device ='cpu', dropout_rate = 0.1):
        super(CRF, self).__init__()
        
        self.id_to_tag = id_to_tag
            
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True)

        
        # output layer
        self.feedforward = nn.Linear(in_features= self.encoder.config.hidden_size, out_features= output_size)

        self.crf_layer = ConditionalRandomField(num_tags= output_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))

        self.dropout = nn.Dropout(dropout_rate)

        self.device = device

    def update_mask(self, labels, attention_mask):

        shape = labels.size()
        labelsU = torch.flatten(labels)
        attention_maskU = torch.flatten(attention_mask)
        idx = (labelsU == -100).nonzero(as_tuple=False)
        idx = torch.flatten(idx)
        labelsU[idx] = torch.tensor(3)
        attention_maskU[idx] = torch.tensor(0)
        labelsU = labelsU.reshape(shape)
        attention_maskU = attention_maskU.reshape(shape)

        return labelsU, attention_maskU
    
    def forward(self, input_ids, attention_mask, labels):

        batch_size = input_ids.size(0)
        
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        
        #for baseline
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        token_scores = self.feedforward(embedded_text_input)
        
        token_scores = F.log_softmax(token_scores, dim=-1)
        

        labelsU, attention_maskU = self.update_mask(copy.deepcopy(labels), copy.deepcopy(attention_mask))
        
        loss = -self.crf_layer(token_scores, labelsU, attention_maskU) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, attention_maskU)

        preds = torch.full(labels.size(),-100) 

        for i in range(batch_size):
            idx,_ = best_path[i]
            preds[i][:len(idx)] = torch.tensor(idx)

        # print(preds.size())
        # print(labels.size())
        
        return {'loss':loss, 'logits':preds.to(self.device)}

    
