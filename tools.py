import pandas as pd
import os



PATH = "prescription"
CSIZE = 10**6

def fillColumns(df,n1,n2):
    df.iloc[:,n1] = df.iloc[:,n1].fillna(df.iloc[:,n2])

def fillAll(df):

    if len(df.columns)<=15: return df

    fillColumns(df,1,15)
    fillColumns(df,14,16)
    fillColumns(df,9,17)
    fillColumns(df,14,18)
    fillColumns(df,10,19)

    df = df.drop(df.iloc[:, 15:], axis=1)

    return df
    
    
def columnCheck(path,csize):

    plist = os.listdir(path)



    for p in plist:

        
        pp = f'{path}//{p}'
        chunk_iter = pd.read_csv(pp,encoding='utf-8 ',chunksize=csize)

        temp = []

        for chunk in chunk_iter:
            
            filled_chunk = fillAll(chunk)
            
            temp.append(filled_chunk)


        result = pd.concat(temp)

        name,_ = os.path.splitext(p)
        result.to_csv(f'{name}_filled.csv',index=False,encoding='utf-8-sig')

    


    

    # for chunk in chunk_iter:
        
        


# print(p)

if __name__ == "__main__":

    columnCheck(PATH,CSIZE)