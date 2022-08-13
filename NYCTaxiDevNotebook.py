# Databricks notebook source
def embedGoogleSlides(deck_id, slide_number=1):

    displayHTML(
        f"""
  <iframe
    src="https://docs.google.com/presentation/d/{deck_id}/embed?slide={slide_number}&rm=minimal"
    frameborder="0"
    width="100%"
    height="700"
  ></iframe>
  """
    )


mlflow = "1c8INezuBa-CohTmUXtjHyG4mksGhic4p9gNXi6IfZ7w"
slide_number = "4"

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Regression Pipeline Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Pipeline on Databricks and inspects its results.
# MAGIC
# MAGIC For more information about the MLflow Regression Pipeline, including usage examples,
# MAGIC see the [Regression Pipeline overview documentation](https://mlflow.org/docs/latest/pipelines.html#regression-pipeline)
# MAGIC and the [Regression Pipeline API documentation](https://mlflow.org/docs/latest/python_api/mlflow.pipelines.html#module-mlflow.pipelines.regression.v1.pipeline).

# COMMAND ----------

embedGoogleSlides(mlflow, 4)

# COMMAND ----------

from mlflow.pipelines import Pipeline

p = Pipeline(profile="databricks")

# COMMAND ----------

p.clean()

# COMMAND ----------

p.inspect()

# COMMAND ----------

p.run("ingest")

# COMMAND ----------

p.run("split")

# COMMAND ----------

p.run("transform")

# COMMAND ----------

p.run("train")

# COMMAND ----------

p.run("evaluate")

# COMMAND ----------

p.run("register")

# COMMAND ----------

p.inspect("train")

# COMMAND ----------

test_data = p.get_artifact("test_data")
test_data.describe()

# COMMAND ----------

trained_model = p.get_artifact("model")
print(trained_model)
