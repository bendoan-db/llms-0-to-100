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

dbutils.library.restartPython()

# COMMAND ----------

from transformers import BertForQuestionAnswering, BertForTokenClassification, BertForSequenceClassification, pipeline, BertTokenizerFast
import torch
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # QA Models

# COMMAND ----------

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased", return_token_type_ids=True)
qa_bert = BertForQuestionAnswering.from_pretrained("bert-large-uncased")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("adversarial_qa", "adversarialQA")

# COMMAND ----------

dataset["train"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

def preprocess(data):
  return(bert_tokenizer(data["question"], data["context"], truncation=True))

processed_dataset=dataset.map(preprocess, batched=True)

# COMMAND ----------

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# COMMAND ----------

from datetime import datetime
path = "/Users/ben.doan@databricks.com/sandbox/llms/bert-qa" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dbutils.fs.mkdirs(path)

# COMMAND ----------

import mlflow
from transformers import Trainer, TrainingArguments
epochs=10
#define training arguments for Trainer, includes deep learning hyper params and checkpointing strategies for large models

mlflow.end_run()

training_args = TrainingArguments(
  output_dir=path,
  num_train_epochs=epochs,
  per_device_train_batch_size=32,
  per_device_eval_batch_size=32,
  load_best_model_at_end=True,

  logging_steps=10,
  log_level="info",
  evaluation_strategy="epoch",
  save_strategy="epoch"
)

# COMMAND ----------

trainer = Trainer(
  model=qa_bert,
  args=training_args,
  train_dataset=processed_dataset["train"],
  eval_dataset=processed_dataset["test"],
  data_collator=data_collator
)

# COMMAND ----------

#test evaluator to make sure everything is running
trainer.evaluate()

# COMMAND ----------

#train model
trainer.train()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-tuned QA Models

# COMMAND ----------

bert_qa = pipeline(task="question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", revision="v1.0")

# COMMAND ----------

sequence = "Who is the best rapper alive?", "Most people say Kendrick Lamar, while others say it is Young Thug"
bert_qa(*sequence)

# COMMAND ----------


