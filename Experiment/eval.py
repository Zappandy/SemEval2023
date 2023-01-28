#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
from torch import cuda
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random
from transformers import DataCollatorForTokenClassification
import evaluate
from util.utils import feval, get_tag_mappings, get_data 
from util.args import create_arg_parser
from util.dataloader import PreDataCollator
os.environ["WANDB_DISABLED"] = "true"




device = 'cuda' if cuda.is_available() else 'cpu'

### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ### Instructions
# 
# Set the variables in the next cell according to the experiment:
# 
# ``LANG``: Set the language. You can find the language codes in the excel file.
# 
# ``TOKENIZER_NAME`` or ``MODEL_NAME``: Huggingface Model link that you used before in training
# 
# ``SET``: Select the dataset that you used before in training
# 
# - ``None`` --> **None Augmentation** (No Augmentation from wiki)
# - ``tags`` --> **Max Augmentation** (Maximum Augmentation from wiki)
# - ``LM`` --> **Entity Extractor** (Augmentation from wiki after extracting tags using other NER model)
# 
# ``CHECKPOINT``: You have the saved models in the output directory. Set the checkpoint you want to evaluate. If you want to evaluate the full trained model, set **Final**
#  
# ``IS_CRF``: True if you want to evaluate the CRF model. Recommended to finish all non-CRF experiments first
# 
# 
# **Don't forget to update the results in the excel sheet. The link is given below.**
# 
# [Link to Excel File](https://docs.google.com/spreadsheets/d/11LXkOBWxpWDGMsi9XC72eMNSJI14Qo2iwP8qugwjyqU/edit#gid=0)

# ### Define Variables

def main():

    args = create_arg_parser()

    LANG = args.language # use None for all lang
    MAX_LEN = 256
    TOKENIZER_NAME = args.model
    MODEL_NAME = args.model
    SET = args.set# or 'tags' or 'LM' or None
    CHECKPOINT = 'Final'
    
    #IS_CRF = args.crf
    IS_CRF = False

    output_dir = f"./output/{MODEL_NAME}-{LANG}-{SET}" if SET!=None else f"./output/{MODEL_NAME}-{LANG}"
    
    
    # ### Preparing data
    
    # In[ ]:
    
    
    # Load data as pandas dataframe
    test_df = get_data(LANG, SET, train=False)
    
    
    if LANG!=None:
        test_df = test_df[test_df['lang']==LANG]
    
    
    # In[ ]:
    
    
    ## Transform into hugginface dataset
    test_df['length'] = test_df.sent.apply(lambda x:len(x.split()))
    test_data = Dataset.from_pandas(test_df)
    
    
    # In[ ]:
    
    
    # Check random data item
    
    print(test_data[6]['sent'])
    print(test_data[6]['labels'])
    
    
    # ### Tokenization
    
    # In[ ]:
    
    
    tags_to_ids, ids_to_tags = get_tag_mappings()
    number_of_labels = len(tags_to_ids)
    
    
    # In[ ]:
    
    
    ## load appropiate tokenizer for pre-trained models
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    
    
    # In[ ]:
    
    
    collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids)
    
    
    # In[ ]:
    
    
    test_tokenized = test_data.map(collator, remove_columns=test_data.column_names, batch_size=4, num_proc=4, batched=True)
    
    
    
    # ### Load Saved Model
    
    # In[ ]:
    
    
    saved_model_dir = f'{output_dir}/checkpoint-{CHECKPOINT}' if CHECKPOINT !='Final' else f'{output_dir}/Final'
    if IS_CRF:
       # model = CRF(MODEL_NAME,ids_to_tags,number_of_labels,device=device)
        checkpoint = torch.load(f"{saved_model_dir}/pytorch_model.bin")
        model.load_state_dict(checkpoint)
    else:
        model = AutoModelForTokenClassification.from_pretrained(saved_model_dir, num_labels=number_of_labels)
    model = model.to(device)
    
    
    # ### Evaluation
    
    # In[ ]:
    
    
    outputs, vis = feval(test_data,test_tokenized, model, device)
    
    
    # In[ ]:
    
    
    print(vis[10])
    
    
    # df = pd.DataFrame(outputs, columns=['sent','predictions','true'])
    
    
    # df.to_csv(f'{output_dir}/outputs.csv',index=False)


if __name__ == '__main__':
    main()
