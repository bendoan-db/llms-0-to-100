# Databricks notebook source
# MAGIC %md 
# MAGIC # Multiheaded Attention

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Bert

# COMMAND ----------

# MAGIC %pip install bertviz

# COMMAND ----------

from transformers import BertModel, BertTokenizer
from bertviz import head_view
import torch
import pandas as pd

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# COMMAND ----------

input_string = "This pasta sauce tastes a little funny."

tokens = tokenizer.encode(input_string)
tensor_inputs = torch.tensor(tokens).unsqueeze(0) #takes tensor shape from (20,) to (1,20)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualize Attention

# COMMAND ----------

layers = model(tensor_inputs, output_attentions=True)
layers

# COMMAND ----------

len(layers)

# COMMAND ----------

attention_layer = layers[2]
final_attention = attention_layer[-1].mean(1)[0]

attention_df = pd.DataFrame(final_attention.detach()).applymap(float).round(3)
attention_df.columns = tokenizer.convert_ids_to_tokens(tokens)
attention_df.index = tokenizer.convert_ids_to_tokens(tokens)

attention_df

# COMMAND ----------

head_view(attention_layer, tokens_list, html_action='return', layer=7, heads=[9])

# COMMAND ----------


