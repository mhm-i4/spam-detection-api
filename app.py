import pickle
from fastapi import FastAPI
import numpy as np
import math,re

app=FastAPI()

with open("spam_ham.pkl","rb") as f:
    data=pickle.load(f)

w=data["w"]
b=data["b"]
N=data["N"]
vocab=data["vocab"]
df=data["df"]

def tokenize(email):
    email=email.lower()
    email=re.sub(r'[^a-zA-Z\s]','',email)
    return email.split()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def vectorize(email):
    vector=np.zeros(len(vocab))
    
    L=tokenize(email)
    
    tf={}
    for word in L:
        if word not in vocab:
            continue
        tf[word]=tf.get(word,0)+1
    
    
    for word in tf:
        if word not in vocab:
            continue
        termfreq=tf[word]/len(L)
        idf=math.log2(N/df[word])
        vector[vocab[word]]=termfreq*idf
        
    return vector

@app.post("/predict")
def predict(email : str):
    vec=vectorize(email)
    x=np.array(vec)
    z=x@w+b
    y_pred=sigmoid(z)
    
    return {
        "spam" : bool(y_pred>0.5),
        "confidence" : float(y_pred)
    }
    





