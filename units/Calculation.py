from re import L
import torch
import torch.nn.functional as F

from units.SentenceEmbedding import embedding_text

def CosSimilarity(trg_embedding,embedding_i):
    trg_vec = torch.FloatTensor(trg_embedding)
    vec_i = torch.FloatTensor(embedding_i)
    cos_values = float(F.cosine_similarity(trg_vec, vec_i, dim=0))
    return cos_values

def CosSimilarityDictionary(trg_text:str,base_dictionary:dict):
    '''
    Cos類似度の計算ユニット
    
    入力:trg_text:対象とする文字列, base_dictionary:比較対象の辞書列
    
    出力:COS類似度を計算した辞書列
    '''
    
    trg_vec = embedding_text(trg_text) #発話文の文書ベクトルの作成
    cosval_dict = {}
    for key_i,value_i in zip(base_dictionary.keys(),base_dictionary.values()):
        cos_i = CosSimilarity(trg_vec,value_i)
        cosval_dict[key_i] = cos_i
    return cosval_dict
    
def Sort_CosDictionary(cos_dict:dict):
    sorted_list = sorted(cos_dict.items(), key = lambda val : val[1], reverse=True)
    return sorted_list

def LargestCosValueKey(sorted_list):
    print(sorted_list)
    res_key = sorted_list[0][0]
    return res_key
