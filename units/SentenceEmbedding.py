from units.SentenceBERTModel import SentenceBertJapanese

MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
model = SentenceBertJapanese(MODEL_NAME)

def embedding_dict(dict_data):
    '''
    辞書型配列の文書を埋め込み，同じキーの辞書型として返す関数
    '''
    print('embedding dictionary ...')
    keys = dict_data.keys()
    values = dict_data.values()
    rtn_dict = {k: model.encode(v, batch_size=8)[0] for k, v in zip(keys, values)}
    return rtn_dict

def embedding_text(text):
    '''
    文字列型の文書を埋め込み，同じ文字列型として返す関数
    '''
    rtn_text = model.encode(text, batch_size=8)[0]
    return rtn_text

    