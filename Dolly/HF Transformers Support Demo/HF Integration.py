# Databricks notebook source
import mlflow
import transformers

architecture = "databricks/dolly-v2-3b"

dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=dolly,
        artifact_path="dolly3b",
        input_example="Hello, Dolly!",
    )

loaded_dolly = mlflow.transformers.load_model(
model_info.model_uri, 
max_new_tokens=250,
)

# COMMAND ----------

import transformers
import mlflow

chat_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
    transformers_model=chat_pipeline,
    artifact_path="chatbot",
    input_example="Hi there!"
  )

# Load as interactive pyfunc
chatbot = mlflow.pyfunc.load_model(model_info.model_uri)

# COMMAND ----------

chatbot.predict("How can I speed up LLM training?")

# COMMAND ----------


