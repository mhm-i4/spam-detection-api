import numpy as np
import math
import pickle
import re
def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def tokenize(email):
    email = email.lower()
    email = re.sub(r'[^a-z0-9\s]', '', email)
    return email.split()

def term_freq(L):
    tf={}
    for word in L:
        tf[word]=tf.get(word,0)+1
    return tf

def vectorize(vocab,L,df,tf,N):
    vec=np.zeros(len(vocab)) # feature vector
    
    for word in tf:
        if word not in vocab:
            continue
        term_freq = tf[word] / len(L)
        idf = math.log2(N / df[word])
        vec[vocab[word]]=term_freq*idf
    
    return vec

    
corpus = [
# ---------------- SPAM ----------------
"free mobile phone offer limited time",
"win cash now click here",
"claim your lottery prize today",
"exclusive deal buy one get one free",
"urgent you won a free vacation",
"limited offer free gift card",
"congratulations you have been selected",
"claim your reward now click link",
"free entry in lucky draw",
"special discount available today",
"win big prizes instantly",
"get rich quick scheme join now",
"cheap medicines available online",
"earn money from home easily",
"click here to claim your bonus",
"you have won a prize claim now",
"free coupons available hurry up",
"limited time sale grab now",
"lowest price guaranteed buy today",
"act now to receive your reward",
"double your income fast",
"investment opportunity high returns",
"win iphone free entry",
"free netflix subscription offer",
"urgent account update required click now",

# ---------------- HAM ----------------
"are you free for meeting tomorrow",
"let us schedule the project call",
"please review the attached report",
"meeting rescheduled to 3 pm",
"can we discuss the assignment",
"see you at lunch today",
"team meeting at 10 am",
"project deadline is tomorrow",
"please send the documents",
"let us finalize the presentation",
"can you share the notes",
"i will call you later",
"thanks for your help",
"please confirm your availability",
"we need to update the report",
"schedule a call with the client",
"let us meet tomorrow morning",
"did you complete the task",
"send me the latest file",
"we have a meeting today",
"please check the email attachment",
"let us discuss the requirements",
"i will be late today",
"thanks for attending the meeting",
"can you review this code"
]

y = np.array([
# spam = 1
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,

# ham = 0
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0
]) #spam , not spam
N = len(corpus)
vocab={}
index=0
for email in corpus:
    for word in tokenize(email):
        if word not in vocab:
            vocab.setdefault(word,index)
            index+=1

#bag of words is literally the frequency of eahc word in vocab across documents -1 way of vectorization
#tfidf is giving more importance to rare words
#tf = freq * idf= log(N/df)

df={}

#CALC DOCUMENT FREQUENCY DF
for email in corpus:
    L=set(tokenize(email))
    for word in L:
        df[word] = df.get(word,0) + 1
                
tf_idf_vec=[]

for email in corpus:
    tf={}
    L=tokenize(email)
    tf=term_freq(L)
    tf_idf_vec.append(vectorize(vocab,L,df,tf,N))

x=np.array(tf_idf_vec)
# shape = no of mails x vocab length
# 3 x 10

n_samples,n_features=x.shape
epoch=1000
w=np.zeros(n_features)
b=0.0
lr=0.01

for _ in range(epoch):
    #forward pass
    z = x @ w + b #y output
    y_p = sigmoid(z)
    error=y_p - y
    #backpropagation
    dw = (x.T @ error)/n_samples
    db = np.mean(error)
    w-= lr*dw
    b-= lr*db

model_data={
    "w":w,
    "b":b,
    "vocab":vocab,
    "df":df,
    "N":N
}
with open("spam_ham.pkl","wb") as f:
    pickle.dump(model_data,f)
