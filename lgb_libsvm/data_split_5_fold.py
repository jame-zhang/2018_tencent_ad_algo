import pandas as pd
import datetime
import gc


start_time = datetime.datetime.now()
print(start_time)

path = r"../../data/data_split/"

for i in range(4):
    print("------"+str(i)+" fold on the way!-----")
    df = pd.DataFrame()
    for j in range(1,11):
        print(str(i*10+j)+"read!")
        df = pd.concat([df, pd.read_csv(path+"train_"+str(i*10+j)+".csv")])
    df.to_csv('train_5fold_'+str(i)+".csv",index=False)

df = pd.DataFrame()

for i in range(1,5):
    df = pd.concat([df, pd.read_csv(path+"train_"+str(40+i)+".csv")])
df =  pd.concat([df, pd.read_csv(path+"train_final.csv")])
df.to_csv("train_5fold_5.csv",index=False)

end_time = datetime.datetime.now()
print((end_time - start_time).seconds)