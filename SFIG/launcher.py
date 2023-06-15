from operator import index
import pandas as pd
import numpy as np
import subprocess


if __name__ =='__main__':
    test_data = pd.read_csv("./data/SWAT/SWaT_Dataset_Attack_v1.csv")
    df_split = np.array_split(test_data, 100)
    i = 0
    for frame in df_split:
        path = "./data/SWAT/test_split_"+str(i)+".csv"
        df = pd.DataFrame(frame)
        attacks= df.loc[df['NormalAttack']== 'Attack', :]
        if attacks.empty:
            preds = pd.DataFrame([0]*len(df), columns=['result'])
            preds.to_csv('./data/predictions_split'+str(i)+'.csv')
        else:
            df.to_csv(path, index=False)
            pid = subprocess.Popen(['python', 'whitebox_attack.py', '-p', path, '-s', str(i)])
        i +=1

