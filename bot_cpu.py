# -*- coding: utf-8 -*-
print("Running bot_cpu")
bot_name = 'bot_cpu'

import sys
try:
    mother_folder = sys.argv[1] ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1
except:
    mother_folder = r"C:\Users\user\Desktop\test1" ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1

sys.path.append(f"{mother_folder}/{bot_name}")  ## 指到自己的Folder
import setting_config

test = setting_config.test

## SQL 位址
env = setting_config.env
server = setting_config.DB_info['address']
database = ''
username = setting_config.DB_info['uid']  
password = setting_config.DB_info['pwd']

## ip
host_ip = setting_config.ip
port_ip = setting_config.port

## Others
thresh_dic = {'SimilarityFAQBot':0, 'SimilarityIntentionBot':0, 'ChatBotKind': 0, '4000Faq': 0, '4000Intention': 0, '4000Emo': 0}

import os
font = f"{mother_folder}/{bot_name}/Models/JhengHei.ttf"

try:
    if not os.path.exists(f'{mother_folder}/{bot_name}/Models'):
        os.makedirs(f'{mother_folder}/{bot_name}/Models')
except:
    pass

import math
e = math.e

'''
!pip install transformers
!pip install tensorflow-gpu
!pip install cuda-python
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

    strategy = tf.distribute.MirroredStrategy(["GPU:0"])
except:
    pass
'''
import pyodbc
import json
import pandas as pd
from opencc import OpenCC
import tensorflow as tf

from transformers import BertTokenizerFast, AutoModelForSequenceClassification

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
''' #1
print(torch.cuda.is_available())
TFbert_model_faq = TFBertForSequenceClassification.from_pretrained('bert-base-chinese')
bert_model_faq = BertForSequenceClassification.from_pretrained('bert-base-chinese')
'''
import random
import time
import jieba

'''
1. load SQL data
'''
def load_sql_data(faq_min, faq_max, intent_min, intent_max, emo_min, emo_max):
    from joblib import Parallel, delayed
    global train_faq, train_intent, train_emo
    global train_synonym, replace_synonyms, train_exclude, replace_exclude, thresh_dic
    def get_target_len(df, col, left, right):
        return df[(df[col].str.len() >= left) & (df[col].str.len() <= right)]
    
    ## 1. traindata
    cnxn = pyodbc.connect(f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={server}; DATABASE={database};UID={username};PWD={password}')
    data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
    
    ## build synonym replace
    train_synonym = pd.read_sql(f"SELECT * FROM {data['SynowordDetail']}", cnxn)

    category_to_first_word = train_synonym.groupby('SynonymId')['Text'].first().to_dict()
    train_synonym['first_word'] = train_synonym['SynonymId'].map(category_to_first_word)

    def replace_synonyms(sentence, synonym, synonym_first=""):
        if type(synonym_first)== str:
            for i in synonym:
                sentence = sentence.replace(i, synonym_first)
        else:
            for i,j in zip(synonym, synonym_first):
                sentence = sentence.replace(i,j)
        return sentence
    
    ## build exclude replace
    train_exclude = pd.read_sql(f"SELECT * FROM {data['SynowordExclude']}", cnxn)

    def replace_exclude(sentence, synonym):
        text = ""
        for j in [i for i in jieba.cut(sentence) if i not in list(synonym)]:
            text+= j
        return text
    
    ## FAQ
    train_faq = pd.read_sql(f"SELECT * FROM {data['SynophaseDetail_bot']}", cnxn)
    
    train_faq['Text'] = [replace_synonyms(i, train_synonym['Text'], train_synonym['first_word']) for i in train_faq['Text']]
    train_faq['Text'] = Parallel(n_jobs=8)(delayed(replace_exclude)(i, train_exclude['Text']) for i in train_faq['Text'])  
    train_faq = get_target_len(train_faq, 'Text', faq_min, train_faq['Text'].str.len().quantile(faq_max))
    train_faq = train_faq[~train_faq['Text'].duplicated()]

    ## Intent
    train_intent = pd.read_sql(f"SELECT * FROM {data['synophaseDetail']}", cnxn)
    
    train_intent = train_intent[~train_intent.Rank.apply(lambda x : x==2)]
    train_intent['Text'] = Parallel(n_jobs=8)(delayed(replace_exclude)(i, train_exclude['Text']) for i in train_intent['Text'])
    train_intent = get_target_len(train_intent, 'Text', intent_min, train_intent['Text'].str.len().quantile(intent_max))
    train_intent = train_intent[~train_intent['Text'].duplicated()]
    
    ## Emo
    train_emo = pd.read_sql(f"SELECT * FROM {data['emotionmodelupdate']}", cnxn)
        
    stp = OpenCC('s2twp')
    train_emo = train_emo.dropna(subset = ['EmotionName'])
    train_emo['Question'] = train_emo['Question'].apply(lambda x :stp.convert(x))
    train_emo['EmotionName'] = train_emo['EmotionName'].apply(lambda x :stp.convert(x))
    train_emo = get_target_len(train_emo, 'Question', emo_min, train_emo['Question'].str.len().quantile(emo_max))
    train_emo = train_emo[~train_emo['Question'].apply(lambda x : x == '')]
    train_emo = train_emo[~train_emo['Question'].duplicated()]
    
    def get_random_rows(group, row_n=1000):
        return group.sample(n=min(row_n, group.shape[0]), random_state=42)
    second_most = train_emo.groupby('EmotionName').size().nlargest().iloc[1]
    train_emo = train_emo.groupby('EmotionName').apply(get_random_rows, second_most).reset_index(drop=True)
    
    for i in list(set(train_emo['EmotionName'])):
        try:
            lack = train_emo.loc[train_emo['EmotionName']==i].sample(n=max(0, second_most-len(train_emo.loc[train_emo['EmotionName']==i])))
        except:
            lack = train_emo.loc[train_emo['EmotionName']==i]
        train_emo = pd.concat([train_emo, lack], ignore_index=True)
    
    ## Threshold
    threshold = pd.read_sql(f"SELECT * FROM {data['Option']}", cnxn)
    for i in thresh_dic.keys():
        thresh_dic[i]= float(threshold['Value'][threshold['Key']==i].values[0])    
    
    ## now_model
    try:
        now_model = pd.read_sql(f"SELECT * FROM {data['BotModel']}", cnxn)
    
        for i in ['4000Faq', '4000Intention', '4000Emo']:
            thresh_dic[i] = now_model['Name'][now_model['Id']==int(thresh_dic[i])].values[0]
            
    except:
        for i,j in zip(['4000Faq', '4000Intention', '4000Emo'], ['gpt2', 'gpt2', 'albert']):
            thresh_dic[i] = j
    
    print("輸入資料:完成")
    
def load_sql_ip():
    global host_ip, port_ip, chat_ip
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server}; \
                          SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))
    
    ## IP
    ip = pd.read_sql(f"SELECT * FROM {data['BotIp']}", cnxn)
                
    host_ip = ip['IP'][ip['Name']=='IP'].values[0]
    port_ip = ip['Port'][ip['Name']=='IP'].values[0]
    if test:
        chat_ip = '192.168.2.106:4020'
    else:
        chat_ip = ip['IP'][ip['Name']=='Chat'].values[0] + ':' + ip['Port'][ip['Name']=='Chat'].values[0]
    

'''
Build dic
'''
## 4. build dictionary
def set_random_seed():
    random.seed(420)
    tf.random.set_seed(420)
    np.random.seed(420)
    torch.manual_seed(420)

def build_dic():
    global train_faq, train_intent, train_emo
    global maxlen_faq, maxlen_intent, maxlen_emo, dic_faq, dic_intent, dic_emo, subword_encoder_faq, subword_encoder_intent, subword_encoder_emo
    # global vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, x_train_intent, x_train_emo, y_train_faq, y_train_intent, y_train_emo
    ## FAQ_train_data
    df_train_faq = tf.data.Dataset.from_tensor_slices(train_faq['Text'])
    subword_encoder_faq = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (i.numpy() for i in tf.data.Dataset.from_tensor_slices(train_faq['Text'])), 
        target_vocab_size=2**13,
        max_subword_length=1)

    vocab_size_faq = subword_encoder_faq.vocab_size

    encode_train_faq = [subword_encoder_faq.encode(i.numpy()) for i in df_train_faq]
    x_train_faq = keras.preprocessing.sequence.pad_sequences(encode_train_faq, padding='post')
    maxlen_faq = x_train_faq.shape[1]

    labeler = LabelEncoder()
    train_faq['label_faq'] = labeler.fit_transform(train_faq['FaqSettingId'])
    train_faq.label_faq = pd.Categorical(train_faq.label_faq)
    dic_faq = dict(zip(range(len(labeler.classes_)), labeler.classes_))
    
    y_train_faq = np.array(train_faq.label_faq)

    ## intent_train_data
    df_train_intent = tf.data.Dataset.from_tensor_slices(train_intent['Text'])
    subword_encoder_intent = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (i.numpy() for i in tf.data.Dataset.from_tensor_slices(train_intent['Text'])), 
            target_vocab_size=2**13,
            max_subword_length=1)

    vocab_size_intent = subword_encoder_intent.vocab_size

    encode_train_intent = [subword_encoder_intent.encode(i.numpy()) for i in df_train_intent]
    x_train_intent = keras.preprocessing.sequence.pad_sequences(encode_train_intent, padding='post')
    maxlen_intent = x_train_intent.shape[1]  

    labeler = LabelEncoder()
    train_intent['label_intent'] = labeler.fit_transform(train_intent['IntentionName'])
    train_intent.label_intent = pd.Categorical(train_intent.label_intent)
    dic_intent = dict(zip(range(len(labeler.classes_)), labeler.classes_))

    y_train_intent = np.array(train_intent.label_intent)
        
    ## Emo_train_data
    df_train_emo = tf.data.Dataset.from_tensor_slices(train_emo['Question'])
    subword_encoder_emo = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (i.numpy() for i in tf.data.Dataset.from_tensor_slices(train_emo['Question'])), 
            target_vocab_size=2**13,
            max_subword_length=1)

    vocab_size_emo = subword_encoder_emo.vocab_size

    encode_train_emo = [subword_encoder_emo.encode(i.numpy()) for i in df_train_emo]
    x_train_emo = keras.preprocessing.sequence.pad_sequences(encode_train_emo, padding='post')
    maxlen_emo = x_train_emo.shape[1]

    labeler = LabelEncoder()
    train_emo['label_emo'] = labeler.fit_transform(train_emo['EmotionName'])
    train_emo.label = pd.Categorical(train_emo.label_emo)
    dic_emo = dict(zip(range(len(labeler.classes_)), labeler.classes_))

    y_train_emo = np.array(train_emo.label_emo)
    print("訓練用資料:完成")
    
    return vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo


'''
2. Transformer model
'''
d_model_T = 64
num_heads = 16
dff = 64

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads = num_heads, key_dim = d_model)
        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim = vocab_size, output_dim = d_model)
        self.pos_emb = layers.Embedding(input_dim = maxlen, output_dim = d_model)
        
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit = maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def run_transformer(func_name, epoch, vocab_size, dic_len, x_train, y_train, test = 0):
    global d_model_T, num_heads, dff
    batchsize = int(len(x_train)**(1/e))
    maxlen = globals()[f'maxlen_{func_name}']
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, d_model_T)
    
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(d_model_T, num_heads, dff)
    x = transformer_block(x)
    x = layers.Dropout(0.1)(x)
    
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(60, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(dic_len, activation="softmax")(x)
    
    transformer_model = keras.Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    if test==1: ## train test split
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        transformer_model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch, validation_data=(x_test, y_test))
    elif test==0: ## train
        transformer_model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch)
    elif test==2:
        transformer_model.load_weights(f"{mother_folder}/{bot_name}/Models/transformer_{func_name}.weights.h5")
        
    try:
        os.remove(f"{mother_folder}/{bot_name}/Models/transformer_{func_name}.weights.h5")
    except:
        pass
    transformer_model.save_weights(f"{mother_folder}/{bot_name}/Models/transformer_{func_name}.weights.h5")
    print(f"transformer_{func_name} model:訓練完畢")
    return transformer_model

'''
load_sql_data(2, 0.7, 2, 0.7, 4, 0.8)
vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo= build_dic()

now_faq = run_transformer('faq', 10, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq, 1)
now_intent = run_transformer('intent', 5, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent, 1)
now_emo = run_transformer('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo, 1)

gen_label('刷卡')
'''

'''
3. RNN
'''
d_model_R = 256

def run_rnn(func_name, epoch, vocab_size, dic_len, x_train, y_train, test=0):
    global d_model_R
    batchsize = int(len(x_train)**(1/e))
    maxlen = globals()[f'maxlen_{func_name}']
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, d_model_R)
    
    x = embedding_layer(inputs)
    x = layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = layers.Dense(60, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(dic_len, activation="softmax")(x)
    
    rnn_model = keras.Model(inputs=inputs, outputs=outputs)
    rnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    if test==1:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        rnn_model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch, validation_data=(x_test, y_test))
    elif test==0:
        rnn_model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch)
    elif test==2:
        rnn_model.load_weights(f"{mother_folder}/{bot_name}/Models/rnn_{func_name}.weights.h5")
        
    try:
        os.remove(f"{mother_folder}/{bot_name}/Models/rnn_{func_name}.weights.h5")
    except:
        pass
    rnn_model.save_weights(f"{mother_folder}/{bot_name}/Models/rnn_{func_name}.weights.h5")
    print(f"rnn_{func_name} model:訓練完畢")
    return rnn_model

'''
load_sql_data(2, 0.7, 2, 0.7, 4, 0.8)
vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo= build_dic()  
    
now_faq = run_rnn('faq', 16, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq, 1)
now_intent = run_rnn('intent',8, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent, 1)
now_emo = run_rnn('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo, 1)

gen_label('有問題')
'''

'''
4. albert & gpt2
'''
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
tokenizer.padding_side = "left"

def encode_Bert(bert_tokenizer, datax, datay, len_n):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_l = []
    token_type_ids_l = []
    attention_mask_l = []
    label_l = []
    for i,j in zip(datax, datay):
        bert_input = bert_tokenizer.encode_plus(i, add_special_tokens=True, 
                                                max_length=len_n, 
                                                padding='max_length', 
                                                truncation=True, 
                                                return_attention_mask=True)
        input_ids_l.append(bert_input['input_ids'])
        token_type_ids_l.append(bert_input['token_type_ids'])
        attention_mask_l.append(bert_input['attention_mask'])
        label_l.append(j)
      
    return tf.data.Dataset.from_tensor_slices((input_ids_l, attention_mask_l, token_type_ids_l, label_l))

## Set predictions
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def get_predictions(model, data, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for i in data:
            tokens_tensors, masks_tensors, segments_tensors = [torch.tensor(j.numpy(), dtype=torch.long) for j in i[:3]]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            prob = logits
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = torch.tensor(i[3].numpy(), dtype=torch.long)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
                probability = prob
            else:
                predictions = torch.cat((predictions, pred))
                probability = torch.cat((probability, prob))
    
    if compute_acc:
        acc = correct / total
        predictions = predictions.tolist()
        probability = list(map(lambda x : softmax(x), probability.tolist()))
        return predictions, acc, probability
    
    predictions = predictions.tolist()
    probability = list(map(lambda x : softmax(x), probability.tolist()))
    return predictions, probability

def run_bert(now_model, bert_tokenizer, bert_model, func_name, epoch, x_train, y_train, test=0):
    batchsize = int(len(x_train)**(1/e))
    bert_model.config.pad_token_id = bert_model.config.eos_token_id
    bert_model.train()
    learning_rate= np.linspace(1e-4, 1e-6, epoch)
    
    maxlen = globals()[f'maxlen_{func_name}']
    if test==1:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        BertTrain = encode_Bert(bert_tokenizer, x_train, y_train, maxlen).shuffle(6000).batch(batchsize)
        BertTest = encode_Bert(bert_tokenizer, x_test, y_test, maxlen).batch(batchsize)
        
        for i,l in zip(range(epoch), learning_rate):
            optimizer = torch.optim.Adam(bert_model.parameters(), lr=l)
            time_start = time.time()
            running_loss1, running_loss2 = 0.0, 0.0
                           
            for j in BertTrain:
                tokens_tensor, masks_tensor, segments_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long) for k in j[:4]]  
                optimizer.zero_grad()
                outputs = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss1 += loss.item()
                
            for j in BertTest:
                tokens_tensor, masks_tensor, segments_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long) for k in j[:4]]
                outputs = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                running_loss2 += loss.item()
            
            _, acc1,_ = get_predictions(bert_model, BertTrain, compute_acc=True)
            _, acc2,_ = get_predictions(bert_model, BertTest, compute_acc=True)
                
            print(f'[train_{func_name} {i+1}] loss: {round(running_loss1, 4)}, acc: {round(acc1, 4)}')
            print(f'[test_{func_name} {i+1}] loss: {round(running_loss2, 4)}, acc: {round(acc2, 4)}')
            
            time_end = time.time()
            time_c= time_end - time_start
            
            print("------------------------------------")
            print('time cost', time_c, 's')
            print('time cost hours', time_c/3600, 'hr')
            print("                                    ")
    elif test==0:
        BertTrain = encode_Bert(bert_tokenizer, x_train, y_train, maxlen).shuffle(6000).batch(batchsize)
        
        for i,l in zip(range(epoch), learning_rate):
            optimizer = torch.optim.Adam(bert_model.parameters(), lr=l)
            time_start = time.time()
            running_loss1, running_loss2 = 0.0, 0.0
                           
            for j in BertTrain:
                tokens_tensor, masks_tensor, segments_tensor, label = [torch.tensor(k.numpy(), dtype=torch.long) for k in j[:4]]  
                optimizer.zero_grad()
                outputs = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensor, attention_mask=masks_tensor, labels=label)
                
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss1 += loss.item()
            
            _, acc1,_ = get_predictions(bert_model, BertTrain, compute_acc=True)
                
            print(f'[train_{func_name} {i+1}] loss: {round(running_loss1, 4)}, acc: {round(acc1, 4)}')
            
            time_end = time.time()
            time_c= time_end - time_start
            
            print("------------------------------------")
            print('time cost', time_c, 's')
            print('time cost hours', time_c/3600, 'hr')
            print("                                    ")
    elif test==2:
        bert_model = torch.load(f"{mother_folder}/{bot_name}/Models/{now_model}_tiny_{func_name}", map_location=torch.device('cpu'))
    
    try:
        os.remove(f"{mother_folder}/{bot_name}/Models/{now_model}_tiny_{func_name}")
    except:
        pass
    
    with open(f"{mother_folder}/{bot_name}/Models/{now_model}_tiny_{func_name}", mode="wb") as f:
        torch.save(bert_model, f)
        
    print(f"{now_model}_tiny_{func_name} model:訓練完畢")
    return bert_model

'''
load_sql_data(2, 0.8, 2, 0.8, 4, 0.8)
vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo= build_dic()
      
now_faq = run_bert('albert', tokenizer, AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_faq)), 
                   'faq', 12, train_faq['Text'], train_faq.label_faq, 1)
now_intent = run_bert('albert', tokenizer , AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_intent)),
                      'intent', 5, train_intent['Text'], train_intent.label_intent, 1)
now_emo = run_bert('albert', tokenizer , AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_emo)), 
                   'emo', 5, train_emo['Question'], train_emo.label_emo, 1)

now_faq = run_bert('gpt2', tokenizer, AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_faq)), 
                   'faq', 12, train_faq['Text'], train_faq.label_faq, 1)
now_intent = run_bert('gpt2', tokenizer , AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_intent)),
                      'intent', 9, train_intent['Text'], train_intent.label_intent, 1)
now_emo = run_bert('gpt2', tokenizer , AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_emo)), 
                   'emo', 8, train_emo['Question'], train_emo.label_emo, 1)

gen_label("刷卡")
'''


'''
6. Emotion_bot
'''
from snownlp import SnowNLP
## lightGBM

def gen_data(data_q, data_emo, data_label):
    global now_emo
    data_q, data_emo, data_label = list(data_q), list(data_emo), list(data_label)
    cols = len(dic_emo)+1 ## SnowNLP have 1 values
    temp = pd.DataFrame(columns = ['Question', 'EmotionName', 'label_emo']+[str(i) for i in range(cols)])
    
    for i,j,k in zip(data_q, data_emo, data_label):
        temp_list = [i, j, k]
        
        try: ## transformer, rnn
            set_random_seed()
            t_emo = keras.preprocessing.sequence.pad_sequences([subword_encoder_emo.encode(i)], maxlen=maxlen_emo, padding='post')
            t_emo = now_emo.predict(t_emo)[0]
        except: ## albert, gpt2
            set_random_seed()
            _, b_emo = get_predictions(now_emo, encode_Bert(tokenizer, [i], [""], maxlen_emo).batch(60))
            t_emo =b_emo[0]

        [temp_list.append(j) for j in t_emo]
        temp_list.append(SnowNLP(i).sentiments)
        
        temp.loc[len(temp.index)] = temp_list
    return temp

## Emo_train_data
def get_emo_data():
    global lgb_model
    try:
        check = pd.read_csv(f"{mother_folder}/{bot_name}/Models/{thresh_dic['4000Emo']}_emo_csv.csv", index_col=0)
        if len(dic_emo)+1+3 == len(check.columns): ## SnowNLP 1, Question EmotionName label 3
            same = check[['Question', 'EmotionName', 'label_emo']].merge(train_emo[['Question', 'EmotionName', 'label_emo']], 
                                                                         on=['Question', 'EmotionName', 'label_emo'], how='inner')
            filtered_check = check[check['Question'].isin(same['Question'])]
            
            diff = train_emo[['Question', 'EmotionName', 'label_emo']][~train_emo[['Question', 'EmotionName', 'label_emo']]['Question'].isin(same['Question'])]
            if len(diff):
                train_emo_csv = pd.concat([gen_data(diff.Question, diff.EmotionName, diff.label_emo), filtered_check], ignore_index=True)
                train_emo_csv.to_csv(f"{mother_folder}/{bot_name}/Models/{thresh_dic['4000Emo']}_emo_csv.csv")
            else:
                print('Emotion have no new data')
        else:
            train_emo_csv = gen_data(train_emo.Question, train_emo.EmotionName, train_emo.label_emo)
            train_emo_csv.to_csv(f"{mother_folder}/{bot_name}/Models/{thresh_dic['4000Emo']}_emo_csv.csv")
    except:
        train_emo_csv = gen_data(train_emo.Question, train_emo.EmotionName, train_emo.label_emo)
        train_emo_csv.to_csv(f"{mother_folder}/{bot_name}/Models/{thresh_dic['4000Emo']}_emo_csv.csv")
    
    train_emo_csv = pd.read_csv(f"{mother_folder}/{bot_name}/Models/{thresh_dic['4000Emo']}_emo_csv.csv", index_col=0)
    
    ## lgb_train
    target = ['Question', 'EmotionName', 'label_emo']
    ##[target.append(str(i)) for i in range(5,9)]
    x_train_emo_lgb = train_emo_csv.drop(columns= target)
    y_train_emo_lgb = train_emo_csv.label_emo

    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(x_train_emo_lgb, y_train_emo_lgb)
    
    print(f"{thresh_dic['4000Emo']}_emo model:訓練完畢")

import string
def gen_emo_label(text, balance=0, gen_data=0):
    global lgb_model, now_emo
    text = text.translate(str.maketrans('', '', string.punctuation))
    set_random_seed()
    temp_list = []
    
    try: ## transformer, rnn
        set_random_seed()
        t_emo = keras.preprocessing.sequence.pad_sequences([subword_encoder_emo.encode(text)], maxlen=maxlen_emo, padding='post')
        t_emo = now_emo.predict(t_emo)[0]
    except: ## albert, gpt2
        set_random_seed()
        _, b_emo = get_predictions(now_emo, encode_Bert(tokenizer, [text], [""], maxlen_emo).batch(60))
        t_emo =b_emo[0]
    
    [temp_list.append(j) for j in t_emo]
    temp_list.append(SnowNLP(text).sentiments)
    
    set_random_seed()
    lgb_aws = lgb_model.predict([temp_list, temp_list], raw_score=True)[0]
    if balance:
        lgb_aws = lgb_aws - balance
    
    if gen_data:
        return lgb_aws
    else:
        return dic_emo[np.argsort(lgb_aws)[-1]]
'''
get_emo_data()
gen_emo_label('你們網站超難用')
'''

'''
7. Word Cloud
'''
from collections import Counter
def return_words(target, word_len, aws_ng=0.1, amount=10):
    stopwords = list(train_exclude['Text'])
    words = [word for sentence in [jieba.cut_for_search(text) for text in target] for word in sentence if word not in stopwords and len(word) == word_len]
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(amount)]
      
    return top_words

from wordcloud import WordCloud
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


'''
8. Algorithm
'''
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def softmax_trans(arr):
    arr = arr* (max(arr)- (sum(arr)/len(arr)))
    return softmax(arr)

def gen_label(text):
    global now_faq, now_intent, now_emo
    try: ## transformer, rnn
        t_faq = keras.preprocessing.sequence.pad_sequences([subword_encoder_faq.encode(text)], maxlen=maxlen_faq, padding='post')
        set_random_seed()
        t_faq = now_faq.predict(t_faq)[0]
    except: ## albert, gpt2
        set_random_seed()
        _, b_faq = get_predictions(now_faq, encode_Bert(tokenizer, [text], [""], maxlen_faq).batch(60))
        t_faq =b_faq[0]
        
    try: ## transformer, rnn
        t_intent = keras.preprocessing.sequence.pad_sequences([subword_encoder_intent.encode(text)], maxlen=maxlen_intent, padding='post')
        set_random_seed()
        t_intent = now_intent.predict(t_intent)[0]
    except: ## albert, gpt2
        set_random_seed()
        _, b_intent = get_predictions(now_intent, encode_Bert(tokenizer, [text], [""], maxlen_intent).batch(60))
        t_intent =b_intent[0]
        
    faqid = [int(dic_faq[i]) for i in np.argsort(t_faq)[::-1][:4]]
    faq_similarity = [float(i) for i in np.sort(t_faq)[::-1][:4]]
    faq = [list(train_faq.loc[train_faq['FaqSettingId']==i, 'Demand'])[0] for i in faqid]
    faq_intent = [list(train_intent.loc[train_intent['FaqSettingId']==i, 'IntentionName'])[0] for i in faqid]
    faq_answer = list(train_faq.loc[train_faq['FaqSettingId']==faqid[0], 'Supply'])[0]
    
    intent = dic_intent[np.argsort(t_intent)[::-1][0]]
    intent_similarity = [float(i) for i in np.sort(t_intent)[::-1][:4]]
    
    return faqid, faq_similarity, faq, faq_intent, faq_answer, intent, intent_similarity

def gen_old_label(text):
    global now_faq, now_intent, now_emo
    try: ## transformer, rnn
        t_faq = keras.preprocessing.sequence.pad_sequences([subword_encoder_faq.encode(text)], maxlen=maxlen_faq, padding='post')
        set_random_seed()
        t_faq = now_faq.predict(t_faq)[0]
    except: ## albert, gpt2
        set_random_seed()
        _, b_faq = get_predictions(now_faq, encode_Bert(tokenizer, [text], [""], maxlen_faq).batch(60))
        t_faq =b_faq[0]

    try: ## transformer, rnn
        t_intent = keras.preprocessing.sequence.pad_sequences([subword_encoder_intent.encode(text)], maxlen=maxlen_intent, padding='post')
        set_random_seed()
        t_intent = now_intent.predict(t_intent)[0]
    except: ## albert, gpt2
        set_random_seed()
        _, b_intent = get_predictions(now_intent, encode_Bert(tokenizer, [text], [""], maxlen_intent).batch(60))
        t_intent =b_intent[0]
        
    referral_faqid = [int(dic_faq[i]) for i in np.argsort(t_faq)[::-1][:4]]
    similarity_faq = [float(i) for i in np.sort(t_faq)[::-1][:4]]
    referral = [list(train_faq.loc[train_faq['FaqSettingId']==i, 'Demand'])[0] for i in referral_faqid]
    faq_answer = list(train_faq.loc[train_faq['FaqSettingId']==referral_faqid[0], 'Supply'])[0]
    
    referral_intent = [list(train_intent.loc[train_intent['FaqSettingId']==i, 'IntentionName'])[0] for i in referral_faqid]
    similarity_intent = [float(i) for i in np.sort(t_intent)[::-1][:3]]
    intent = dic_intent[np.argsort(t_intent)[::-1][0]]
            
    return referral_intent, intent, referral_faqid[1:4], referral_faqid[0], referral, similarity_faq, similarity_intent, faq_answer

def additional_cal_faq(n_list):
    return [pow(i*100, 0.5)/10 for i in n_list]
    
def additional_cal_intent(n_list):
    temp = sum([pow(i, 0.5) for i in n_list])
    return [i/temp for i in n_list]

def trans_float(n_list, r=3):
    return [round(i, r) for i in n_list]


'''
9. API APP
'''
import requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
app = FastAPI()
print("API:Ready to GO")

@app.get("/")
async def hello():
    return 'Hello, I am bot_cpu'

@app.get("/updatefaq")
async def reload_data():
    load_sql_data(2, 0.8, 2, 0.8, 4, 0.8)
    return thresh_dic

@app.get("/now_model")
async def set_now_model():
    global thresh_dic
    return thresh_dic

@app.get("/word_cloud")
async def word_cloud():
    try:
        final = ' '.join([' '.join(return_words(train_emo['Question'], i)) for i in [2,3,4]])
        word = WordCloud(width=600, height=300, prefer_horizontal=1, collocations=False, min_font_size=10, colormap="tab10", font_path=font).generate(final)
        plt.figure(figsize=(20,10), facecolor='k')
        plt.imshow(word, interpolation ="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f'{mother_folder}/{bot_name}/Models/word.png', format='png', facecolor='k', bbox_inches='tight')
        plt.close()
        return f'Image saved in {mother_folder}/{bot_name}/Models/word.png'
    except:
        return 'reload_data first'

@app.get("/updatemodel")
async def train_model():
    global now_faq, now_intent, now_emo
    load_sql_data(2, 0.8, 2, 0.8, 4, 0.8)
    vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo= build_dic()
    
    if thresh_dic['4000Faq']== 'transformer':
        now_faq = run_transformer('faq', 10, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq)
    elif thresh_dic['4000Faq']== 'rnn':
        now_faq = run_rnn('faq', 16, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq)
    elif thresh_dic['4000Faq']== 'albert':        
        now_faq = run_bert('albert', tokenizer, 
                           AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_faq)), 
                           'faq', 12, train_faq['Text'], train_faq.label_faq)
    elif thresh_dic['4000Faq']=='gpt2':
        now_faq = run_bert('gpt2', tokenizer, 
                           AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_faq)), 
                           'faq', 12, train_faq['Text'], train_faq.label_faq)
        
    if thresh_dic['4000Intention']== 'transformer':
        now_intent = run_transformer('intent', 7, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent)
    elif thresh_dic['4000Intention']== 'rnn':
        now_intent = run_rnn('intent', 8, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent)
    elif thresh_dic['4000Intention']== 'albert':        
        now_intent = run_bert('albert', tokenizer , 
                              AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_intent)),
                              'intent', 5, train_intent['Text'], train_intent.label_intent)
    elif thresh_dic['4000Intention']=='gpt2':
        now_intent = run_bert('gpt2', tokenizer , 
                              AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_intent)),
                              'intent', 9, train_intent['Text'], train_intent.label_intent)
        
    if thresh_dic['4000Emo']== 'transformer':
        now_emo = run_transformer('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo)
    elif thresh_dic['4000Emo']== 'rnn':
        now_emo = run_rnn('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo)        
    elif thresh_dic['4000Emo']== 'albert':        
        now_emo = run_bert('albert', tokenizer , 
                           AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_emo)), 
                           'emo', 5, train_emo['Question'], train_emo.label_emo)
    elif thresh_dic['4000Emo']=='gpt2':
        now_emo = run_bert('gpt2', tokenizer , 
                           AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_emo)), 
                           'emo', 8, train_emo['Question'], train_emo.label_emo)
    
    get_emo_data()
    gen_label('有問題')
    gen_emo_label('瀏覽器無法打開')    
    
    return f"{[[i, thresh_dic[i]] for i in ['4000Faq', '4000Intention', '4000Emo']]} done"

@app.get("/load_model")
async def load_model():
    global now_faq, now_intent, now_emo
    load_sql_data(2, 0.8, 2, 0.8, 4, 0.8)
    vocab_size_faq, vocab_size_intent, vocab_size_emo, x_train_faq, y_train_faq, x_train_intent, y_train_intent, x_train_emo, y_train_emo= build_dic()
    
    temp = []
    try:
        if thresh_dic['4000Faq']== 'transformer':
            now_faq = run_transformer('faq', 10, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq, 2)
        elif thresh_dic['4000Faq']== 'rnn':
            now_faq = run_rnn('faq', 16, vocab_size_faq, len(dic_faq), x_train_faq, y_train_faq, 2)
        elif thresh_dic['4000Faq']== 'albert':        
            now_faq = run_bert('albert', tokenizer, 
                               AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_faq)), 
                               'faq', 12, train_faq['Text'], train_faq.label_faq, 2)
        elif thresh_dic['4000Faq']=='gpt2':
            now_faq = run_bert('gpt2', tokenizer, 
                               AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_faq)), 
                               'faq', 12, train_faq['Text'], train_faq.label_faq, 2)
        temp.append(f"Faq {thresh_dic['4000Faq']} model loaded.")
    except:
        temp.append(f"Faq {thresh_dic['4000Faq']} model not exist, please train first.")
    
    try:
        if thresh_dic['4000Intention']== 'transformer':
            now_intent = run_transformer('intent', 7, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent, 2)
        elif thresh_dic['4000Intention']== 'rnn':
            now_intent = run_rnn('intent', 8, vocab_size_intent, len(dic_intent), x_train_intent, y_train_intent, 2)
        elif thresh_dic['4000Intention']== 'albert':        
            now_intent = run_bert('albert', tokenizer , 
                                  AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_intent)),
                                  'intent', 5, train_intent['Text'], train_intent.label_intent, 2)
        elif thresh_dic['4000Intention']=='gpt2':
            now_intent = run_bert('gpt2', tokenizer , 
                                  AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_intent)),
                                  'intent', 9, train_intent['Text'], train_intent.label_intent, 2)
        temp.append(f"Intent {thresh_dic['4000Intention']} model loaded.")
    except:
        temp.append(f"Intent {thresh_dic['4000Intention']} model not exist, please train first.")
            
    try:
        if thresh_dic['4000Emo']== 'transformer':
            now_emo = run_transformer('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo, 2)
        elif thresh_dic['4000Emo']== 'rnn':
            now_emo = run_rnn('emo', 9, vocab_size_emo, len(dic_emo), x_train_emo, y_train_emo, 2)        
        elif thresh_dic['4000Emo']== 'albert':        
            now_emo = run_bert('albert', tokenizer , 
                               AutoModelForSequenceClassification.from_pretrained('ckiplab/albert-tiny-chinese', num_labels=len(dic_emo)), 
                               'emo', 5, train_emo['Question'], train_emo.label_emo, 2)
        elif thresh_dic['4000Emo']=='gpt2':
            now_emo = run_bert('gpt2', tokenizer , 
                               AutoModelForSequenceClassification.from_pretrained('ckiplab/gpt2-tiny-chinese', num_labels=len(dic_emo)), 
                               'emo', 8, train_emo['Question'], train_emo.label_emo, 2)
        temp.append(f"Emo {thresh_dic['4000Emo']} model loaded.")
    except:
        temp.append(f"Emo {thresh_dic['4000Emo']} model not exist, please train first.")
    
    get_emo_data()
    gen_label('有問題')
    gen_emo_label('瀏覽器無法打開')    
    
    return temp
    
@app.get('/dscbot')
async def dscbot_get(request: Request):
    global thresh_dic, row_dict
    text = request.query_params.get('text', '')
    try:
        history = request.query_params.get('history', '')
        history = eval(history)
        
        threshold_faq = float(request.query_params.get('threshold_faq', thresh_dic['SimilarityFAQBot']))
        threshold_intent = float(request.query_params.get('threshold_intent', thresh_dic['SimilarityIntentionBot']))
    except:
        pass
    
    version=request.query_params.get('version', '1')
    
    text = replace_synonyms(text, train_synonym['Text'], train_synonym['first_word'])
    emotion = gen_emo_label(text)
    
    if version=='1':
        Referral_intent, intent, Referral_faqid, faqid, Referral, Similarity_faq, Similarity_intent, faq_answer = gen_old_label(text)
        Similarity_faq, Similarity_intent = trans_float(Similarity_faq), trans_float(Similarity_intent)
        
        row_dict = {'Bot' : bot_name,
                        'Answer' : faq_answer,
                        'Question' : Referral[0],
                        'Answer_faqsettingid' : faqid,
                        'Referral' : Referral[1:],
                        'Referral_faqsettingid' : Referral_faqid,
                        'Referral_intent' : Referral_intent[1:],
                        'Similarity' : Similarity_faq,
                        'emotion' : emotion,
                        'intent' : Referral_intent[0],
                        'type' : 1,
                        'end' : 1,
                        'max_prob' : Similarity_faq[0],
                        'pred_intent' : intent,
                        'pred_intent_prob' : Similarity_intent[0]
                        }
        '''
        try:
            if (float(Similarity_faq[0]) < threshold_faq) & (float(Similarity_intent[0]) < threshold_intent):
                target = f"http://{chat_ip}/chat?text={text}"
                r = requests.get(target)
                
                row_dict = {'Bot' : "ChatBot",
                            'Answer' : r.text,
                            'Question' : 0,
                            'Answer_faqsettingid' : 0,
                            'Referral' : 0,
                            'Referral_faqsettingid' : 0,
                            'Referral_intent' : Referral_intent[1:],
                            'Similarity' : 0,
                            'emotion' : emotion,
                            'intent' : 0,
                            'type' : 1,
                            'end' : 1,
                            'max_prob' : Similarity_faq[0],
                            'pred_intent' : intent,
                            'pred_intent_prob' : Similarity_intent[0]
                            }
            else:
                row_dict = {'Bot' : bot_name,
                            'Answer' : faq_answer,
                            'Question' : Referral[0],
                            'Answer_faqsettingid' : faqid,
                            'Referral' : Referral[1:],
                            'Referral_faqsettingid' : Referral_faqid,
                            'Referral_intent' : Referral_intent[1:],
                            'Similarity' : Similarity_faq,
                            'emotion' : emotion,
                            'intent' : Referral_intent[0],
                            'type' : 1,
                            'end' : 1,
                            'max_prob' : Similarity_faq[0],
                            'pred_intent' : intent,
                            'pred_intent_prob' : Similarity_intent[0]
                            }
        except:
            row_dict = {'Bot' : "ChatBot",
                        'Answer' : "聊天小幫手忙碌中",
                        'Question' : 0,
                        'Answer_faqsettingid' : 0,
                        'Referral' : 0,
                        'Referral_faqsettingid' : 0,
                        'Referral_intent' : Referral_intent[1:],
                        'Similarity' : 0,
                        'emotion' : emotion,
                        'intent' : 0,
                        'type' : 1,
                        'end' : 1,
                        'max_prob' : Similarity_faq[0],
                        'pred_intent' : intent,
                        'pred_intent_prob' : Similarity_intent[0]
                        }
            '''
    elif version=='2':
        faqid, faq_similarity, faq, faq_intent, faq_answer, intent, intent_similarity = gen_label(text)
        faq_similarity, intent_similarity = trans_float(faq_similarity), trans_float(intent_similarity)
        
        try:
            if (float(Similarity_faq[0]) < threshold_faq) & (float(Similarity_intent[0]) < threshold_intent):
                target = f"http://{chat_ip}/chat?text={text}"
                r = requests.get(target)
                
                row_dict = {'bot' : "ChatBot",
                            'answer' : r.text,
                            'demand' : {'faqId':0, 'similarity':faq_similarity[0], 'text':0, 'intent':faq_intent[0]},
                            'referral' : [{'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[1]},
                                          {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[2]},
                                          {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[3]}],
                            'emotion' : emotion,
                            'type' : 1,
                            'end' : 1,
                            'predIntent' : intent,
                            'predIntentProb' : intent_similarity[0]
                            }
            else:
                row_dict = {'bot' : bot_name,
                            'answer' : faq_answer,
                            'demand' : {'faqId':faqid[0], 'similarity':faq_similarity[0], 'text':faq[0], 'intent':faq_intent[0]},
                            'referral' : [{'faqId':faqid[1], 'similarity':faq_similarity[1], 'text':faq[1], 'intent':faq_intent[1]},
                                          {'faqId':faqid[2], 'similarity':faq_similarity[2], 'text':faq[2], 'intent':faq_intent[2]},
                                          {'faqId':faqid[3], 'similarity':faq_similarity[3], 'text':faq[3], 'intent':faq_intent[3]}],
                            'emotion' : emotion,
                            'type' : 1,
                            'end' : 1,
                            'predIntent' : intent,
                            'predIntentProb' : intent_similarity[0]
                            }
        except:
            row_dict = {'bot' : "ChatBot",
                        'answer' : "聊天小幫手忙碌中",
                        'demand' : {'faqId':0, 'similarity':faq_similarity[0], 'text':0, 'intent':faq_intent[0]},
                        'referral' : [{'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[1]},
                                      {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[2]},
                                      {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[3]}],
                        'emotion' : emotion,
                        'type' : 1,
                        'end' : 1,
                        'predIntent' : intent,
                        'predIntentProb' : intent_similarity[0]
                        }
    return row_dict

@app.post('/dscbot')
async def dscbot_post(request: Request):
    global row_dict
    data = await request.json()
    text = data['text']  
    try:        
        try:
            threshold_faq = float(data['threshold_faq'])
        except:
            threshold_faq = thresh_dic['SimilarityFAQBot']
            
        try:
            threshold_intent = float(data['threshold_intent'])
        except:
            threshold_intent = thresh_dic['SimilarityIntentionBot']
    except:
        pass
    
    try:
        version=data['version']
    except:
        version='1'
        
    text = replace_synonyms(text, train_synonym['Text'], train_synonym['first_word'])
    emotion = gen_emo_label(text)
    
    if version=='1':
        Referral_intent, intent, Referral_faqid, faqid, Referral, Similarity_faq, Similarity_intent, faq_answer = gen_old_label(text)
        Similarity_faq, Similarity_intent = trans_float(Similarity_faq), trans_float(Similarity_intent)
        row_dict = {'Bot' : bot_name,
                    'Answer' : faq_answer,
                    'Question' : Referral[0],
                    'Answer_faqsettingid' : faqid,
                    'Referral' : Referral[1:],
                    'Referral_faqsettingid' : Referral_faqid,
                    'Referral_intent' : Referral_intent[1:],
                    'Similarity' : Similarity_faq,
                    'emotion' : emotion,
                    'intent' : Referral_intent[0],
                    'type' : 1,
                    'end' : 1,
                    'max_prob' : Similarity_faq[0],
                    'pred_intent' : intent,
                    'pred_intent_prob' : Similarity_intent[0]
                    }
        '''
        try:
            if (float(Similarity_faq[0]) < threshold_faq) & (float(Similarity_intent[0]) < threshold_intent):
                target = f"http://{chat_ip}/chat?text={text}"
                r = requests.get(target)
                
                row_dict = {'Bot' : "ChatBot",
                            'Answer' : r.text,
                            'Question' : 0,
                            'Answer_faqsettingid' : 0,
                            'Referral' : 0,
                            'Referral_faqsettingid' : 0,
                            'Referral_intent' : Referral_intent[1:],
                            'Similarity' : 0,
                            'emotion' : emotion,
                            'intent' : 0,
                            'type' : 1,
                            'end' : 1,
                            'max_prob' : Similarity_faq[0],
                            'pred_intent' : intent,
                            'pred_intent_prob' : Similarity_intent[0]
                            }
            else:
                row_dict = {'Bot' : bot_name,
                            'Answer' : faq_answer,
                            'Question' : Referral[0],
                            'Answer_faqsettingid' : faqid,
                            'Referral' : Referral[1:],
                            'Referral_faqsettingid' : Referral_faqid,
                            'Referral_intent' : Referral_intent[1:],
                            'Similarity' : Similarity_faq,
                            'emotion' : emotion,
                            'intent' : Referral_intent[0],
                            'type' : 1,
                            'end' : 1,
                            'max_prob' : Similarity_faq[0],
                            'pred_intent' : intent,
                            'pred_intent_prob' : Similarity_intent[0]
                            }
        except:
            row_dict = {'Bot' : "ChatBot",
                        'Answer' : "聊天小幫手忙碌中",
                        'Question' : 0,
                        'Answer_faqsettingid' : 0,
                        'Referral' : 0,
                        'Referral_faqsettingid' : 0,
                        'Referral_intent' : Referral_intent[1:],
                        'Similarity' : 0,
                        'emotion' : emotion,
                        'intent' : 0,
                        'type' : 1,
                        'end' : 1,
                        'max_prob' : Similarity_faq[0],
                        'pred_intent' : intent,
                        'pred_intent_prob' : Similarity_intent[0]
                        }
            '''
    elif version=='2':
        faqid, faq_similarity, faq, faq_intent, faq_answer, intent, intent_similarity = gen_label(text)
        faq_similarity, intent_similarity = trans_float(faq_similarity), trans_float(intent_similarity)
        
        try:
            if (float(Similarity_faq[0]) < threshold_faq) & (float(Similarity_intent[0]) < threshold_intent):
                target = f"http://{chat_ip}/chat?text={text}"
                r = requests.get(target)
                
                row_dict = {'bot' : "ChatBot",
                            'answer' : r.text,
                            'demand' : {'faqId':0, 'similarity':faq_similarity[0], 'text':0, 'intent':faq_intent[0]},
                            'referral' : [{'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[1]},
                                          {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[2]},
                                          {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[3]}],
                            'emotion' : emotion,
                            'type' : 1,
                            'end' : 1,
                            'predIntent' : intent,
                            'predIntentProb' : intent_similarity[0]
                            }
            else:
                row_dict = {'bot' : bot_name,
                            'answer' : faq_answer,
                            'demand' : {'faqId':faqid[0], 'similarity':faq_similarity[0], 'text':faq[0], 'intent':faq_intent[0]},
                            'referral' : [{'faqId':faqid[1], 'similarity':faq_similarity[1], 'text':faq[1], 'intent':faq_intent[1]},
                                          {'faqId':faqid[2], 'similarity':faq_similarity[2], 'text':faq[2], 'intent':faq_intent[2]},
                                          {'faqId':faqid[3], 'similarity':faq_similarity[3], 'text':faq[3], 'intent':faq_intent[3]}],
                            'emotion' : emotion,
                            'type' : 1,
                            'end' : 1,
                            'predIntent' : intent,
                            'predIntentProb' : intent_similarity[0]
                            }
        except:
            row_dict = {'bot' : "ChatBot",
                        'answer' : "聊天小幫手忙碌中",
                        'demand' : {'faqId':0, 'similarity':faq_similarity[0], 'text':0, 'intent':faq_intent[0]},
                        'referral' : [{'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[1]},
                                      {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[2]},
                                      {'faqId':0, 'similarity':0, 'text':0, 'intent':faq_intent[3]}],
                        'emotion' : emotion,
                        'type' : 1,
                        'end' : 1,
                        'predIntent' : intent,
                        'predIntentProb' : intent_similarity[0]
                        }
    return row_dict
    
        
import uvicorn
if test:
    uvicorn.run(app, host='192.168.2.209', port=4500)
else:
    uvicorn.run(app, host=host_ip, port=port_ip)

