import torch
import numpy as np
import torch.nn.functional as F

def softmax(x):
    y = np.exp(x)
    f_x = y / np.sum(np.exp(x))
    return f_x
#句子词列表
sent=np.array([1,2,3,4,1])
#词向量embedding
embedding_dict=dict([(1,[0.1,0.2,0.3,0.4]),(2,[0.9,.1,.3,.4]),(3,[.3,.4,.5,.6]),(4,[.5,.6,.7,.8])])
#句子长度
len_sent=len(sent)
embedding_dim=len(embedding_dict[1])
print(embedding_dim)
sent_embedding=[]
for i in sent:
    sent_embedding.append(embedding_dict[i])
print(np.array(sent_embedding))
#(batch,句子长度，词向量维度)
sent_embedding=np.array(sent_embedding).reshape(-1,len_sent,embedding_dim)#(1,5,4)

# Wq=np.linspace(0,len_sent*embedding_dim-1,num=len_sent*embedding_dim).reshape((len_sent,embedding_dim))
# Wk=np.linspace(len_sent*embedding_dim,2*len_sent*embedding_dim,num=len_sent*embedding_dim).reshape((len_sent,embedding_dim))
# Wv=np.linspace(2*len_sent*embedding_dim,3*len_sent*embedding_dim,num=len_sent*embedding_dim).reshape((len_sent,embedding_dim))
# Wq=np.linspace(0,embedding_dim-1,num=embedding_dim).reshape((-1,embedding_dim))
# Wk=np.linspace(embedding_dim,2*embedding_dim-1,num=embedding_dim).reshape((-1,embedding_dim))
# Wv=np.linspace(2*embedding_dim,3*embedding_dim-1,num=embedding_dim).reshape((-1,embedding_dim))
Wq=np.random.rand(1,embedding_dim)
Wk=np.random.rand(1,embedding_dim)
Wv=np.random.rand(1,embedding_dim)

# for i in range(len_sent):
#     ai=sent_embedding[0][i]
#     qi=np.matmul(Wq,ai)
#     ki=np.matmul(Wk,ai)
#     alpha1i=np.matmul(qi.reshape((-1,len_sent),ki)[0]
emb_mat=sent_embedding.transpose(0,2,1)[0]
q_vec=np.squeeze(np.matmul(Wq,emb_mat))
k_vec=np.squeeze(np.matmul(Wk,emb_mat))
v_vec=np.squeeze(np.matmul(Wv,emb_mat))
b_output=[]

for i in range(len(q_vec)):
    alpha_res=softmax(q_vec[i]*k_vec)
    b_vec=np.zeros(embedding_dim)
    for j in range(len(alpha_res)):
        b_vec=b_vec+alpha_res[j]*sent_embedding[0][j]
    b_output.append(b_vec)
        
b_mat=np.array(b_output)
# print(softmax([0.21850275, 0.89326652, 0.7530602 , 1.60816864, 0.21850275]))
# src=torch.Tensor([0.21850275, 0.89326652, 0.7530602 , 1.60816864, 0.21850275])
# print(F.softmax(src))
# --------------------------------------------------------
b_vec=np.multiply(alpha_res,v_vec)
Wq.shape
emb_mat.shape
sent_embedding[0].shape
sent_embedding.shape
