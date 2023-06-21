# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-large and DeepSpeed
# MAGIC
# MAGIC This proceeds just like the tuning of `t5-small`, but requires a larger instance type such as the `g5.12xlarge` on AWS (4 x A10 GPUs), and a smaller batch size per device.
# MAGIC This also changes below to use DeepSpeed, so it must be installed along with `accelerate` too:

# COMMAND ----------

# MAGIC %pip install 'transformers==4.29.0' datasets evaluate rouge-score git+https://github.com/microsoft/DeepSpeed accelerate 

# COMMAND ----------

# MAGIC %pip install 'accelerate>=0.19.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import transformers
print(transformers.__version__)

import torch
print(torch.__version__)

# COMMAND ----------

import os
import json

host = json.loads(dbutils.notebook.entry_point.getDbutils().notebook() \
  .getContext().toJson())['tags']['browserHostName']

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + host
os.environ['TRANSFORMERS_CACHE'] = "/dbfs/tmp/ben.doan@databricks.com/cache/hf"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/ben.doan@databricks.com/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC Here the `python` command to run the script has become `deepspeed`, and `--deepspeed` supplies additional configuration:
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC     "fp16": {
# MAGIC         "enabled": false
# MAGIC     },
# MAGIC
# MAGIC     "bf16": {
# MAGIC       "enabled": true
# MAGIC     },
# MAGIC
# MAGIC     "scheduler": {
# MAGIC         "type": "WarmupLR",
# MAGIC         "params": {
# MAGIC             "warmup_min_lr": "auto",
# MAGIC             "warmup_max_lr": "auto",
# MAGIC             "warmup_num_steps": "auto"
# MAGIC         }
# MAGIC     },
# MAGIC
# MAGIC     "zero_optimization": {
# MAGIC         "stage": 2,
# MAGIC         "offload_optimizer": {
# MAGIC             "device": "none"
# MAGIC         },
# MAGIC         "allgather_partitions": true,
# MAGIC         "allgather_bucket_size": 2e8,
# MAGIC         "overlap_comm": true,
# MAGIC         "reduce_scatter": true,
# MAGIC         "reduce_bucket_size": 2e8,
# MAGIC         "contiguous_gradients": true
# MAGIC     },
# MAGIC
# MAGIC     "gradient_accumulation_steps": "auto",
# MAGIC     "gradient_clipping": "auto",
# MAGIC     "steps_per_print": 2000,
# MAGIC     "train_batch_size": "auto",
# MAGIC     "train_micro_batch_size_per_gpu": "auto",
# MAGIC     "wall_clock_breakdown": false
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && deepspeed \
# MAGIC     /Workspace/Repos/ben.doan@databricks.com/transformers/examples/pytorch/summarization/run_summarization.py \
# MAGIC     --deepspeed /Workspace/Repos/ben.doan@databricks.com/transformers/examples/pytorch/summarization/ds_config_zero2_no_offload_adafactor.json \
# MAGIC     --model_name_or_path t5-large \
# MAGIC     --do_train \
# MAGIC     --do_eval \
# MAGIC     --train_file /dbfs/tmp/ben.doan@databricks.com/review/camera_reviews_train.csv \
# MAGIC     --validation_file /dbfs/tmp/ben.doan@databricks.com/review/camera_reviews_val.csv \
# MAGIC     --source_prefix "summarize: " \
# MAGIC     --output_dir /dbfs/tmp/ben.doan@databricks.com/review/t5-large-summary-ds \
# MAGIC     --overwrite_output_dir \
# MAGIC     --optim adafactor \
# MAGIC     --num_train_epochs 4 \
# MAGIC     --gradient_checkpointing \
# MAGIC     --bf16 \
# MAGIC     --per_device_train_batch_size 8 \
# MAGIC     --per_device_eval_batch_size 8 \
# MAGIC     --predict_with_generate \
# MAGIC     --run_name "t5-large-fine-tune-reviews-ds"

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count, pandas_udf
from transformers import pipeline
import pandas as pd

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/ben.doan@databricks.com/review/t5-large-summary-ds",\
  tokenizer="/dbfs/tmp/ben.doan@databricks.com/review/t5-large-summary-ds", \
  num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load("/tmp/ben.doan@databricks.com/review/cleaned")

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

display(review_by_product_df.select("reviews", "summary").limit(10))

# COMMAND ----------

sample_review = "summarize: " + review_by_product_df.select("reviews").head(1)[0]["reviews"]

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/ben.doan@databricks.com/review/t5-large-summary-ds",\
  tokenizer="/dbfs/tmp/ben.doan@databricks.com/review/t5-large-summary-ds",\
  num_beams=10, min_new_tokens=50, device="cuda:0")

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------


