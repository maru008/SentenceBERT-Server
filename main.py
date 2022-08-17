#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import time 
import sys
import json
import socket

#自作モジュールインポート
from units.SentenceBERTModel import SentenceBertJapanese
from units.SentenceEmbedding import embedding_dict
from units.Calculation import CosSimilarityDictionary,Sort_CosDictionary,LargestCosValueKey


HOST_NAME = "127.0.0.1"
PORT = 8080


sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#localhostとlocal portを指定
sock.bind((HOST_NAME,PORT))

#server動作開始
sock.listen(1)
print("server start!")

#基準データを埋め込み
json_data = open('data/Base_sentence_text.json','r')
base_sentence_dict = json.load(json_data) 
embeddings_dict = embedding_dict(base_sentence_dict)
print("Done!")

#接続を許可して、待つ
client,remote_addr = sock.accept()
print("accepted remote. remote_addr {}.".format(remote_addr))
while True:
    #接続されたら、データが送られてくるまで待つ
    rcv_data = client.recv(1024)
    #接続が切られたら、終了
    if not rcv_data:
        print("receive data don't exist")
        break
    else:
        #入力形式はキー入力文書
        get_dict = dict(json.loads(rcv_data))
        #発話内容の文書
        trg_text = get_dict["text"]
        #比較対象とするキーのリスト
        base_key_ls  = get_dict["keys"]
        
        trg_base_embedding_dict = {key: val for key, val in embeddings_dict.items() if key in base_key_ls}
        trg_Css_embedding_dict = CosSimilarityDictionary(trg_text,trg_base_embedding_dict)
        trg_sorted_cos_dict = Sort_CosDictionary(trg_Css_embedding_dict)
        res_key = LargestCosValueKey(trg_sorted_cos_dict)
        
        #clientにcos類似度が最も高いkeyを送信
        client.sendall(f'{res_key}\n'.encode())

print("close client communication")
#clientとserverのsocketを閉じる
client.close()
sock.close()

