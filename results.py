import pandas as pd
import multiprocessing as mp
import os

path = "prescription"
p_list = os.listdir(path)

def process_file(l, drug):

    npath = f'{path}/{l}'
    print(npath)
    csize = 10**6
    temp = []
    chunk_iter = pd.read_csv(npath, encoding='euc-kr', chunksize=csize)
    for chunk in chunk_iter:
        filtered_chunk = chunk[chunk['약품일반성분명코드'] == drug]
        temp.append(filtered_chunk)
    return pd.concat(temp)

# def dataArrange(df):


    
        




if __name__ == '__main__':

    # 필요한 약물만 추출하고 저장하는 코드

    drugDf = pd.read_csv("DRUG_LIST.csv")

    drugCodes = drugDf['약품일반성분명코드'].tolist()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for drugCode in drugCodes:
            result = pool.starmap(process_file, [(l, drugCode) for l in p_list])
            all = pd.concat(result)

            print(drugDf.loc[drugDf['약품일반성분명코드']==drugCode])
            all.to_csv(f"{drugCode}_prescription.csv", index=False, encoding="utf-8-sig")
            print(all.shape)


# import pandas as pd
# import os

# path = "prescription"
# p_list = os.listdir(path)
# csize = 10**6

# def process_file(l):

#     npath = f'{path}/{l}'
#     return pd.read_csv(npath, encoding='euc-kr')

# if __name__ == '__main__':

#     # 필요한 약물만 추출하고 저장하는 코드

#     drugDf = pd.read_csv("DRUG_LIST.csv")

#     drugCodes = drugDf['약품일반성분명코드'].tolist()
    
#     # 모든 파일을 먼저 병합
#     all_data = pd.concat([process_file(l) for l in p_list])
#     all_data.to_csv("all_prescription.csv",index=False, encoding="utf-8-sig")

#     for drugCode in drugCodes:
#         # 병합된 데이터에서 필요한 약물만 추출
#         filtered_data = all_data[all_data['약품일반성분명코드'] == drugCode]
        
#         print(drugDf.loc[drugDf['약품일반성분명코드']==drugCode])
#         filtered_data.to_csv(f"{drugCode}_prescription.csv", index=False, encoding="utf-8-sig")
#         print(filtered_data.shape)
