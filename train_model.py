# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
#imports das libs
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

import mlflow

#import dos dados

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

type(df)

# COMMAND ----------

#exibe o uso de memoria do df
df.info(memory_usage='deep')

# COMMAND ----------

# DBTITLE 1,Definição das variáveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list(set(df.columns.tolist()) - set([target_column,id_column]))

y = df[target_column]
X = df[features_columns]

X

# COMMAND ----------

# DBTITLE 1,Split Test e Train
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=42)

print("Número de linhas em X_train:",X_train.shape[0])
print("Número de linhas em X_test:",X_test.shape[0])
print("Número de linhas em y_train:",y_train.shape[0])
print("Número de linhas em y_test:",y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setup do Experimento MLFlow
mlflow.set_experiment("/Users/willian.sotocorno@unesp.br/dota-unesp-willian")

# COMMAND ----------

# DBTITLE 1,Run do Experimento
with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    model = ensemble.ExtraTreesClassifier()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    
    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    print("Acuracia em treino:", acc_train)

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print("Acuracia em test:", acc_test)
    

# COMMAND ----------

z
