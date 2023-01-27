#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive/')
# %cd "/content/drive/MyDrive/Colab Notebooks/SemEval2023/Evaluation"


# In[ ]:


#get_ipython().system('pip install transformers')
#get_ipython().system('pip install datasets')
#get_ipython().system('pip install evaluate')
#get_ipython().system('pip install colorama')
#get_ipython().system('pip install wikipedia-api')
#get_ipython().system('pip install sentencepiece')


# In[ ]:


import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
import random
from datasets import Dataset
from util.utils import get_tag_mappings, write_conll_format_preds
from util.dataloader import PreDataCollator
from util.args import create_arg_parser
import nltk
nltk.download('punkt')
os.environ["WANDB_DISABLED"] = "true"
from helper import prepare_data


# In[ ]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[3]:
args = create_arg_parser()

LANG = args.language # use None for all lang
MAX_LEN = 256
TOKENIZER_NAME = 'garNER/%s'%args.model
MODEL_NAME = 'garNER/%s'%args.model
if args.set == 'None':
    set = None
else:
    set = args.set
SET = set # 'LM' or None
EVAL_SET = 'test'


# ## Read Data

# In[9]:


if SET=='LM' :
    filename = f'./Augmented-Dataset/{LANG}-{EVAL_SET}.csv'
    data = pd.read_csv(filename)
    data['length'] = data.sent.apply(lambda x:len(x.split()))
    test_df = data.drop(columns=['sent'])
    test_df = test_df.rename(columns={'augmented_sen':'sent'})
    test_data = Dataset.from_pandas(test_df)
else:
    filename = f'../Dataset/{LANG}-{EVAL_SET}.conll'
    data = prepare_data(filename)
    data['length'] = data.sent.apply(lambda x:len(x.split()))
    test_data = Dataset.from_pandas(data)


# ### Tokenization

# In[10]:


tags_to_ids, ids_to_tags = get_tag_mappings()
number_of_labels = len(tags_to_ids)


# In[11]:


## load appropiate tokenizer for pre-trained models
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids, Set= EVAL_SET)


# In[12]:


test_tokenized = test_data.map(collator, remove_columns=test_data.column_names, batch_size=8, num_proc=8, batched=True)


# ### Load Saved Model

# In[ ]:


model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=number_of_labels)
model = model.to(device)


# In[ ]:


from torch.utils.data import DataLoader
from util.utils import compute_metrics_test
dataloader = DataLoader(test_tokenized, batch_size=args.batch_size)
outputs = []
for batch in tqdm(dataloader):

  inp_ids = torch.stack(batch["input_ids"], axis=1).to(device)
  label_ids = torch.stack(batch["labels"], axis=1).to(device)
  mask = torch.stack(batch["attention_mask"], axis=1).to(device)
  logits = model(input_ids=inp_ids, attention_mask=mask).logits
  pred_ids = torch.argmax(logits, dim=-1)
  for i in range(inp_ids.shape[0]):
      _, predicts = compute_metrics_test(pred_ids[i], label_ids[i])
      pred_tags = [ids_to_tags[idx] for idx in predicts if idx!=-100]
      outputs.append((batch['ID'][i],batch['sents'][i], pred_tags))


# ### Evaluation

# In[ ]:


predictions = pd.DataFrame(outputs, columns=['ID','sent','predictions'])
predictions.head()


# In[ ]:


if EVAL_SET!='test':
  from operator import add
  from functools import reduce
  true = [label.strip().split() for label in test_data['labels']]
  preds = predictions.predictions.array
  predictions['true'] = true
  preds = reduce(add, preds)
  true = reduce(add, true)
  from sklearn.metrics import classification_report
  print(classification_report(preds, true, output_dict=True)['macro avg']['f1-score'])


# In[ ]:


import os

dir = f'./{LANG}/{EVAL_SET}'
if not os.path.exists(f'./{LANG}'):
    os.makedirs(f'./{LANG}')
if not os.path.exists(dir):
    os.makedirs(dir)


# In[ ]:


predictions['predictions'] = predictions['predictions'].apply(lambda x: " ".join(x))


# In[ ]:


filename = MODEL_NAME.split('/')[-1]
fileConll = f'{dir}/{filename}.pred.conll'
write_conll_format_preds(fileConll, predictions, col='predictions')


# In[ ]:


fileCsv = f'{dir}/outputs-{filename}.csv'
predictions.to_csv(fileCsv,index=False)

