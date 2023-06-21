# Databricks notebook source
# MAGIC %pip install einops

# COMMAND ----------

# MAGIC %pip install torch==2.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import torch
print("mlflow version:", mlflow.__version__)
print("torch version:", torch.__version__)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
falcon = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = falcon(
   "Generate a single multiple choice question about the solar system. Question should have 3 options.",
    max_length=2000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------

base_dir = "/dbfs/Users/ben.doan@databricks.com/LLMs/artifacts/"
dbutils.fs.mkdirs("/dbfs/Users/ben.doan@databricks.com/LLMs/artifacts/")

# COMMAND ----------

base_dir+model

# COMMAND ----------

import mlflow
import transformers

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=falcon,
        artifact_path="tiiuae/falcon-7b-instruct",
        input_example="Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    )

loaded_dolly = mlflow.transformers.load_model(
model_info.model_uri, 
max_new_tokens=250,
)

# COMMAND ----------


