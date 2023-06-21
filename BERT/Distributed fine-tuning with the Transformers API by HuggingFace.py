# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Distributed fine-tuning with the Hugging Face Transformers API
# MAGIC
# MAGIC This notebook provides an example of how to fine-tune a Hugging Face model using the [Transformers API](https://huggingface.co/blog/pytorch-ddp-accelerate-transformers) and the [TorchDistributor API](https://github.com/apache/spark/blob/master/python/pyspark/ml/torch/distributor.py). The fine-tuning guidance in this article is based on this [Hugging Face blog post](https://huggingface.co/blog/sentiment-analysis-python#a-fine-tuning-model-with-python).
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - (Recommended) GPU instances

# COMMAND ----------

# MAGIC %md ## Define the number of GPUs to use
# MAGIC
# MAGIC In this example you use a cluster with 4 worker nodes. If you are using a different cluster configuration, update `NUM_WORKERS` accordingly.

# COMMAND ----------

import torch

NUM_WORKERS = 4

def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()

NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]
USE_GPU = NUM_GPUS_PER_WORKER > 0

# COMMAND ----------

# MAGIC %md ## Preprocess your data
# MAGIC
# MAGIC Initialize the tokenizer and collator for preprocessing the data.

# COMMAND ----------

import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# COMMAND ----------

# MAGIC %md ### Import and preprocess the IMDB dataset
# MAGIC
# MAGIC One key difference in the Hugging Face blog and this notebook is that this example uses **all** of the IMDB data, not just 3000 data points.

# COMMAND ----------

from datasets import load_dataset # import the huggingface Datasets option
import pandas as pd

imdb = load_dataset("imdb")
train = pd.DataFrame(imdb["train"])
test = pd.DataFrame(imdb["test"])

texts = train["text"].tolist()
labels = train["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2)

train_encodings = tokenizer(train_texts, truncation=True)
val_encodings = tokenizer(val_texts, truncation=True)

class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenized_train = ImdbDataset(train_encodings, train_labels)
tokenized_test = ImdbDataset(val_encodings, val_labels)

# COMMAND ----------

# MAGIC %md ## Set up the training function
# MAGIC
# MAGIC The TorchDistributor API has support for single node multi-GPU training as well as multi-node training. 
# MAGIC
# MAGIC When you wrap the single-node code in the `train()` function, Databricks recommends you include all the import statements inside the train() function to avoid library pickling issues. You can return any picklable object in `train_model()`, but that means you can't return `Trainer` since that can't be picklable without a process group. You can instead return the best checkpoint path and use that externally. 

# COMMAND ----------

import numpy as np
from datasets import load_metric

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}
 
output_dir = "/dbfs/rithwik-db/imdb/finetuning-sentiment-model-v1" # Save to DBFS (required)

def train_model():
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
      output_dir=output_dir,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=2,
      weight_decay=0.01,
      save_strategy="epoch",
      report_to=[], # REMOVE MLFLOW INTEGRATION FOR NOW
      push_to_hub=False,  # DO NOT PUSH TO MODEL HUB FOR NOW,
      load_best_model_at_end=True, # RECOMMENDED
      metric_for_best_model="eval_loss", # RECOMMENDED
      evaluation_strategy="epoch" # RECOMMENDED
    )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_test,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.state.best_model_checkpoint

# It is recommended to create a separate local trainer from pretrained model instead of using the trainer used in distributed training
def test_model(ckpt_path):
  model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
  local_trainer = Trainer(model=model,eval_dataset=tokenized_test,tokenizer=tokenizer,data_collator=data_collator,compute_metrics=compute_metrics)
  return local_trainer.evaluate()

def test_example(ckpt_path, inputs):
  from transformers import pipeline
  model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
  p = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
  outputs = p(inputs)
  return ["Positive" if item["label"] == "LABEL_0" else "Negative" for item in outputs]

# COMMAND ----------

# MAGIC %md ## Run local training

# COMMAND ----------

single_node_ckpt_path = train_model()

# COMMAND ----------

test_model(single_node_ckpt_path)

# COMMAND ----------

# MAGIC %md ## Run distributed training on a single node with multiple GPUs
# MAGIC
# MAGIC Distributor with `local_mode=True` will run the `train()` function directly on the driver node of the spark cluster.
# MAGIC
# MAGIC To configure how many GPUs to use in total for this run, pass `num_processes=N` to the Distributor where `N` is the number of GPUs you want to use on the driver node. Note that you don't need to actually make any changes to your training code.

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

NUM_PROCESSES = torch.cuda.device_count()
print(f"We're using {NUM_PROCESSES} GPUs")
single_node_multi_gpu_ckpt_path = TorchDistributor(num_processes=NUM_PROCESSES, local_mode=True, use_gpu=USE_GPU).run(train_model)

# COMMAND ----------

test_model(single_node_multi_gpu_ckpt_path)

# COMMAND ----------

# MAGIC %md ### Run distributed training on multi-node
# MAGIC
# MAGIC The TorchDistributor with `local_mode=False` (default) runa the `train()` function on the worker nodes of the Spark cluster.
# MAGIC
# MAGIC To configure the number of GPUs to use in total for this run, pass `num_processes=TOTAL_NUM_GPUS` to the TorchDistributor. Notice that you don't need to actually make any changes to your training code.

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

NUM_PROCESSES = NUM_GPUS_PER_WORKER * NUM_WORKERS
print(f"We're using {NUM_PROCESSES} GPUs")
multi_node_ckpt_path = TorchDistributor(num_processes=NUM_PROCESSES, local_mode=False, use_gpu=USE_GPU).run(train_model)

# COMMAND ----------

test_model(multi_node_ckpt_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the model with Transformers's `pipeline` API

# COMMAND ----------

def test_example(ckpt_path, inputs):
  from transformers import pipeline
  model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=2)
  p = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
  outputs = p(inputs)
  return [{i:"Positive"} if item["label"] == "LABEL_1" else {i:"Negative"} for i, item in zip(inputs, outputs)]

test_example(multi_node_ckpt_path, ["i love this movie", "this movie sucks!"])
