#%%
import numpy as np
import pandas as pd
import Calculation as c
#%%
choice = int(input("enter 1 to load complete data or 2 to load non empty data : "))
df = pd.read_csv("train.csv") if choice == 1 else pd.read_csv("nonEmptyTrain.csv")
directory = f"NonEmpty" if choice == 2 else f"complete"
# %%
def numeralizeColumn(column,dropValue = False):
    global df
    uniVals = df[column].unique()
    Uniques = dict(zip(uniVals,range(1,len(uniVals)+1)))
    df[f"Numerical{column}"] = [Uniques[key] for key in df[column]]
    df = df.drop(columns=[column]) if dropValue == True else df

def fillNan(column,newColumn=True):
    global df
    avg = df[column].mean()
    temp = [avg if pd.isnull(val) == True else val for val in df[column]]
    if newColumn == False :
        df[column] = temp
    else:
        df[f"filled{column}"] = temp

# %%
df = df.drop(columns=["Name","PassengerId","Ticket"])
numeralizeColumn("Sex",dropValue=True)
numeralizeColumn("Embarked",dropValue=True)
numeralizeColumn("Cabin",dropValue=True)
fillNan("Age",newColumn=False) if choice == 1 else None
# %%
Y = np.array([list(df["Survived"])])
df = df.drop(columns=["Survived"])
df["xNaught"] = [1]*(df.shape[0])
df = df[[df.columns[-1]]+ list(df.columns[:-1])]
#%%
model = c.Model(df,Y,directory)    
# %%

# %%

# %%
