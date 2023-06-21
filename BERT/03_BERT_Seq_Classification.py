# Databricks notebook source
# MAGIC %md
# MAGIC # BERT Finetuning Tasks
# MAGIC 1. Sequence Classification
# MAGIC     - ex. Sentiment Analysis
# MAGIC     - Add pooling + Feed Forward layers to classify sequence
# MAGIC 2. Token Classification
# MAGIC     - ex. Entity Recognition
# MAGIC     - Add pooling + Feed Forward layers to classify every token'
# MAGIC 3. Question and Answer
# MAGIC     - Take 2 sequences, question and context
# MAGIC     - Add Feed Forward + Softmax
# MAGIC     - Outputs probability that token is start of answer and end of answer. Then the tokens can be extracted using those indices to formulate response

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

from transformers import BertForQuestionAnswering, BertForTokenClassification, BertForSequenceClassification, pipeline, BertTokenizer
import torch
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretuned Model: Finbert
# MAGIC Pre-tuned model built on BERT

# COMMAND ----------

finbert = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

# COMMAND ----------

finbert("Nvidia popped off today, rising by 20%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual Fine-tuning

# COMMAND ----------

from transformers import Trainer, TrainingArguments, \
  DistilBertForSequenceClassification, DistilBertTokenizerFast, DataCollatorWithPadding, pipeline
from datasets import load_metric, load_dataset, Dataset
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Loading from HuggingFace 

# COMMAND ----------

#load snips dataset the training split (in this case this dataset only has a training split)
snips_dataset = load_dataset("snips_built_in_intents", split="train")
dataset_features = snips_dataset.features

# COMMAND ----------

#display features 
dataset_features

# COMMAND ----------

#get labels
labels = dataset_features["label"].names
labels

# COMMAND ----------

#split dataset
snips_dataset = snips_dataset.train_test_split(test_size=0.3)

# COMMAND ----------

snips_dataset["train"][0]

# COMMAND ----------

#get total number of labels
total_classes = dataset_features['label'].num_classes
print(total_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing: Tokenization

# COMMAND ----------

#load distillbert tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# COMMAND ----------

#preprocess text dataset into tokens
def preprocess(batch_dataset):
  return tokenizer(batch_dataset["text"], truncation=True)

# COMMAND ----------

#preprocess snips dataset into tokens dataset
snip_tokens = snips_dataset.map(preprocess, batched=True)

# COMMAND ----------

#original dataset
snips_dataset

# COMMAND ----------

#dataset now tokenized (input_ids = tokens) with attention masks
snip_tokens

# COMMAND ----------

#example record
snip_tokens["test"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing: Collation

# COMMAND ----------

#0 pads text based on longest input sequence. Also sets attention score of pads to 0
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#load distilbert seq classification model
sequence_clf_model = DistilBertForSequenceClassification\
  .from_pretrained("distilbert-base-uncased", num_labels=total_classes)

# COMMAND ----------

#create index -> label dictionary and configure model to use text labels instead of label indices
sequence_clf_model.config.id2label = {i: l for i, l in enumerate(labels)}
sequence_clf_model.config.id2label[0]

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Training/Finetuning

# COMMAND ----------

metric = load_metric("accuracy")#load accuracy metric

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1) #get predictions by argmaxing logit output
  
  accuracy_metric = metric.compute(predictions=predictions, references=labels) #calculate accuracy with predictions and labels

  return accuracy_metric

# COMMAND ----------

path = "/Users/ben.doan@databricks.com/sandbox/llms/bert" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dbutils.fs.mkdirs(path)

print(path)

# COMMAND ----------

import mlflow
epochs=10
#define training arguments for Trainer, includes deep learning hyper params and checkpointing strategies for large models

mlflow.end_run()

training_args = TrainingArguments(
  output_dir=path,
  num_train_epochs=epochs,
  per_device_train_batch_size=32,
  per_device_eval_batch_size=32,
  load_best_model_at_end=True,

  warmup_steps=len(snip_tokens["train"]) // 5,
  weight_decay=0.05,

  logging_steps=1,
  log_level="info",
  evaluation_strategy="epoch",
  save_strategy="epoch"
)

# COMMAND ----------

trainer = Trainer(
  model=sequence_clf_model,
  args=training_args,
  train_dataset=snip_tokens["train"],
  eval_dataset=snip_tokens["test"],
  compute_metrics=compute_metrics,
  data_collator=data_collator
)

# COMMAND ----------

#test evaluator to make sure everything is running
trainer.evaluate()

# COMMAND ----------

#train model
trainer.train()

# COMMAND ----------

trainer.evaluate()

# COMMAND ----------

trainer.save_model()

# COMMAND ----------

pipe = pipeline("text-classification", "/Users/ben.doan@databricks.com/sandbox/llms/bert13-06-2023_18-55-13", tokenizer=tokenizer)

# COMMAND ----------

pipe("Will I need a jacket tomorrow in New York?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Token Classification

# COMMAND ----------

bert_tc = pipeline("token-classification", model="2rtl3/mn-bert-base-demo-named-entity", tokenizer="2rtl3/mn-bert-base-demo-named-entity")

# COMMAND ----------

string_input = "My name is Ben and I have a condo in Washington D.C."
bert_tc(string_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA Models

# COMMAND ----------

bert_qa = pipeline(task="question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", revision="v1.0")

# COMMAND ----------

sequence = "Who is the best rapper alive?", "Most people say Kendrick Lamar, while others say it is Young Thug"
bert_qa(*sequence)

# COMMAND ----------


