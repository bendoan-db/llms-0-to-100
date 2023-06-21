# Databricks notebook source
# MAGIC %pip install bertviz

# COMMAND ----------

from transformers import BertModel, BertTokenizer
from bertviz import head_view
import torch
import pandas as pd

# COMMAND ----------

model = BertModel.from_pretrained("bert-base-uncased")

# COMMAND ----------

len(model.encoder.layer) #mbase BERT has 12 encoders in encoder stack

# COMMAND ----------

#inspect first layer
model.encoder.layer[0]

# COMMAND ----------

# DBTITLE 1,Bert Layers
# MAGIC %md
# MAGIC     (self): BertSelfAttention(
# MAGIC       (query): Linear(in_features=768, out_features=768, bias=True)
# MAGIC       (key): Linear(in_features=768, out_features=768, bias=True)
# MAGIC       (value): Linear(in_features=768, out_features=768, bias=True)
# MAGIC       (dropout): Dropout(p=0.1, inplace=False)
# MAGIC     )
# MAGIC
# MAGIC In the attention layer above we have 768 dimensions that will be applied to each token in the query, key, and value inputs. Dropout is used for generalizability and speed

# COMMAND ----------

model.encoder.layer[0]

# COMMAND ----------

config = BertConfig()

# COMMAND ----------


