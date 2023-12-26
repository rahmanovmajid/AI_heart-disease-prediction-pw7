from normalization import z_scaling
import mlp
import pandas as pd

df = pd.read_csv('heart_disease_dataset.csv')
columns_array = list(df.columns)
df_array = df.values
numerical_variables = ['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved', 'num_major_vessels','st_depression']
z_scaling(df_array,columns_array,numerical_variables)

mlp = mlp.MLP(df_array,4)
mlp.train(1000)
mlp.model_evaluation()
