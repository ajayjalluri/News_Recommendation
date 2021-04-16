import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from flask import Flask, render_template, request, url_for

import numpy as np
from flask import jsonify
# initiate flask
app = Flask(__name__)

news = pd.read_csv(r"D:\Desktop\recommendation\news.csv")
news = news.drop(["Unnamed: 0"],axis =1)

user = pd.read_csv("user_vector.csv")
user = user.drop(["Unnamed: 0"],axis =1)
user_m = user.values
doc_embeds = pd.read_csv("article_vector.csv")

doc_embeds = doc_embeds.drop(["Unnamed: 0"],axis =1)
doc_embeds_m = doc_embeds.values

def article_recommend(user_id,user_matrix,doc_embeds): # user id
    user = user_matrix[user_id]
    user=user.reshape((1,50))
    result_cont=cosine_similarity(doc_embeds,user)
    result_users=cosine_similarity(user_matrix,user)
    user_list_col =[]
    user_list_col.append(heapq.nlargest(5, range(len(result_users)), result_users.take))
    collab_vectors=np.sum(doc_embeds[user_list_col[0]],axis=0)
    collab_vectors=(collab_vectors/(len(user_list_col[0])))
    user_list_col.clear()
    collab_vectors=collab_vectors.reshape((1,50))
    result_coll=cosine_similarity(doc_embeds,collab_vectors)
    top_list =[]
    top_list.append(heapq.nlargest(5, range(len(result_cont)), result_cont.take))
    top_list.append(heapq.nlargest(5, range(len(result_coll)), result_coll.take))
    flat_list = [item for sublist in top_list for item in sublist]
    print(flat_list)
    top_list.clear()
    return flat_list


@app.route('/<u>')
def home(u):
    k = int(u)
    l = article_recommend(k,user_m,doc_embeds_m)
    m = []
    for x in l :
        m.append("A"+str(x))
    m = list(set(m))
    d = {}
    for x in m :
       d[x] =[]

    for x in m :

        i = int(x[1:])
        d[x].append(news["1"][i])
        d[x].append(news["3"][i])
        d[x].append(news["5"][i])

    return jsonify(d)

if __name__ == '__main__':
    app.run(debug = False, port = 8000)
