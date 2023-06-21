# Databricks notebook source
# MAGIC %md 
# MAGIC ## Model Fine-tuning Approaches 
# MAGIC <br>
# MAGIC
# MAGIC 1. Update entire model on labeled data, including weights in base model and layers added on top
# MAGIC     - Slowest, best performance
# MAGIC 2. Freeze a subset of the model
# MAGIC     - Average training speed and performance
# MAGIC 3. Freeze the entire model and only train additional layers on top
# MAGIC     - Fastest training, worst performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## HuggingFace Trainer API
# MAGIC - API abstraction to remove the need for dataset splitting, gradient computation, etc.
# MAGIC
# MAGIC ### Key Components
# MAGIC
# MAGIC 1. Dataset (pre-split)
# MAGIC 2. DataCollator: Forms batches of data from Datasets to be fed into model
# MAGIC 3. Training Arguments: Keeps track of trianing arguments, like saving strategy, and learning rate, scheduler parameters
# MAGIC 4. Trainer: API to the pytorch training loop

# COMMAND ----------

# MAGIC %md 
# MAGIC ## BERT Architecture Basics

# COMMAND ----------

from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# COMMAND ----------

named_params = list(model.named_parameters())
print(f"BERT has {str(len(named_params))} different named params. \n")

print("====Embedding Layer====\n")
for p in named_params[0:5]:
  print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print("\n\n====First Encoder====\n")
for p in named_params[5:21]:
  print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print("\n\n====Output Layer====\n")
for p in named_params[-2:]:
  print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# COMMAND ----------

#pooler acts as representation as entire input sentence

# COMMAND ----------

input_String = "Yo this meat sauce tastes a little funny"
input_Embedding = torch.tensor(tokenizer.encode(input_String)).unsqueeze(0)

model_output = model(input_Embedding)
model_output

# COMMAND ----------

model_output.last_hidden_state #embedding for each token in the 12th (last) encoder in the stack

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wordpiece Tokenization

# COMMAND ----------

print(f"Total words in BERT base vocab: {str(len(tokenizer.vocab))}")

# COMMAND ----------

text = "My marinara sauce tastes a little funny. Why is this?"
tokens = tokenizer.encode_plus(text) #also displays attention mask, which is a binary feature to determine if token should be included in attention calculation
print(tokens)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contextual Representation

# COMMAND ----------

python_string_1 = tokenizer.encode("I love my pet python")
python_string_2 = tokenizer.encode("I love coding in python")

# COMMAND ----------

python_pet_embedding = model(torch.tensor(python_string_1).unsqueeze(0))[0][:,5,:].detach().numpy()
python_programming_embedding = model(torch.tensor(python_string_2).unsqueeze(0))[0][:,5,:].detach().numpy()

# COMMAND ----------

#embedding for the strings are different
python_pet_embedding

# COMMAND ----------

python_programming_embedding

# COMMAND ----------


