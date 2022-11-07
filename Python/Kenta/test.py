import pandas as pd

fruit_dict = {
    1: 'apple',
    2: 'banana',
    3:'mango',
    4:'apricot',
    5:'kiwi',
    6:'orange'}

name_dict = {1: 'apple',
             2: 'banana',
             3:'mango',
             4:'apricot',
             5:'kiwi',
             6:'orange'}

dict = {"fruit" : fruit_dict,
        "name" : name_dict}

# df = pd.concat({k: pd.Series(v) for k, v in dict.items()}).reset_index()
df = pd.DataFrame([dict])
print(df)
