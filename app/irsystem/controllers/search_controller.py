from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from collections import Counter, defaultdict
import numpy as np
import os
import math
import pickle
import resource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import concurrent.futures

project_name = "Used Car Recommendations"
net_id = "Ikra Monjur: im324, Yoon Jae Oh: yo82, Fareeza Hasan: fh244, Destiny Malloy: dam359, David Hu: dsh236"

mac_memory_in_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)
print("memory beg", mac_memory_in_MB)


@irsystem.route('/', methods=['GET'])
def search():
    search_bar = request.args.get('search')
    query = search_bar
    transmission = request.args.get('transmission') # if manual or automatic add it to query
    two_doors = request.args.get("2dr")
    four_doors = request.args.get("4dr")
    conv = request.args.get("conv")
    elec = request.args.get("elec")

    if transmission == "automatic" or transmission == "manual":
        query += " " + transmission

    if two_doors == "on":
        query += " 2dr"

    if four_doors == "on":
        query += " 4dr"

    if conv == "on":
        query += " convertible"

    if elec == "on":
        query += " electric hybrid"



    if not query:
        data = []
        output_message = ''
    else:
        output_message = "Your search: " + search_bar
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_ranked, query)
                data = future.result()
        except:
            data = ["No results for current search | Try a new search"]
        # ranking_thread = threading.Thread(target=get_ranked, name="Ranker", args=[query])
        # ranking_thread.start()
        # data = ranking_thread.join()

        if len(data) == 0:
            data = ["No results for current search | Try a new search"]
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


# query --> measure sim with "vehicle title + review" --> get the top five similar things
treebank_tokenizer = TreebankWordTokenizer()

# absolute_path = os.path.dirname(os.path.abspath(__file__))
# file_path = absolute_path + '/new_reviews.json'

# with open(file_path) as json_file:
#     reviews_dict = json.load(json_file)

# total_cars = len(reviews_dict.keys())

# id_to_car = {id:car for id, car in enumerate(reviews_dict)}

# tfidf_vec_reviews = TfidfVectorizer()
# tfidf_mat_reviews = tfidf_vec_reviews.fit_transform([reviews_dict[d]['review'] for d in reviews_dict])
# tfidf_vec_titles = TfidfVectorizer()
# tfidf_mat_titles = tfidf_vec_titles.fit_transform([reviews_dict[d]['title'][0] for d in reviews_dict])
# ratings = [reviews_dict[d]['rating'] for d in reviews_dict]

# Save all of them in pickle files

# pickle.dump(tfidf_vec_reviews, open("tfidf_vec_reviews.pickle", "wb"))
# pickle.dump(tfidf_mat_reviews, open("tfidf_mat_reviews.pickle", "wb"))
# pickle.dump(tfidf_vec_titles, open("tfidf_vec_titles.pickle", "wb"))
# pickle.dump(tfidf_mat_titles, open("tfidf_mat_titles.pickle", "wb"))
# pickle.dump(ratings, open("ratings.pickle", "wb"))
# pickle.dump(id_to_car, open("id_to_car.pickle", "wb"))
# pickle.dump(total_cars, open("total_cars.pickle", "wb"))

# Load all the pickle files

# tfidf_mat_reviews = pickle.load(open("tfidf_mat_reviews.pickle", "rb"))
# tfidf_mat_titles = pickle.load(open("tfidf_mat_titles.pickle", "rb"))
ratings = pickle.load(open("ratings.pickle", "rb"))
id_to_car = pickle.load(open("id_to_car.pickle", "rb"))
total_cars = pickle.load(open("total_cars.pickle", "rb"))
tfidf_vec_reviews = pickle.load(open("tfidf_vec_reviews.pickle", "rb"))
tfidf_vec_titles = pickle.load(open("tfidf_vec_titles.pickle", "rb"))

# U_tit, S_tit, V_T_tit = svds(tfidf_mat_titles.T, k=325)
# pickle.dump(U_tit, open("u_tit_svd.pickle", "wb"))
# pickle.dump(S_tit, open("s_tit_svd.pickle", "wb"))
# pickle.dump(V_T_tit, open("v_t_svd.pickle", "wb"))

# U_rev, S_rev, V_T_rev = svds(tfidf_mat_reviews.T, k=100)
# pickle.dump(U_rev, open("u_rev_svd.pickle", "wb"))
# pickle.dump(S_rev, open("s_rev_svd.pickle", "wb"))
# pickle.dump(V_T_rev, open("v_t_rev_svd.pickle", "wb"))

U_tit = pickle.load(open("u_tit_svd.pickle", "rb"))
S_tit = pickle.load(open("s_tit_svd.pickle", "rb"))
V_T_tit = pickle.load(open("v_t_svd.pickle", "rb"))
U_rev = pickle.load(open("u_rev_svd.pickle", "rb"))
S_rev = pickle.load(open("s_rev_svd.pickle", "rb"))
V_T_rev = pickle.load(open("v_t_rev_svd.pickle", "rb"))

mac_memory_in_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)
print("memory after load", mac_memory_in_MB)

stemmer = SnowballStemmer("english")


def get_ranked(query):
    query = str(query)
    # print("query is ", query)
    sim_mat = np.transpose(cosine_sim(query)).flatten()

    if sim_mat[(np.argsort((sim_mat))[::-1][:5])[0]] == 0.0:
        # print("in if")
        return []
    else:
        # print("in else")
        ranked_lst = [(id_to_car[i], round(ratings[i], 2), round(sim_mat[i], 2))
                      for i in np.argsort(sim_mat)[::-1][:5]]
        # print("list is ", ranked_lst)

        return ranked_lst
    # printing out to see the cosine scores
    #for i in np.argsort((sim_mat))[::-1][:5]:
        #print(sim_mat[i])


# data is the car name

def cosine_sim(query):
    query_tokens = treebank_tokenizer.tokenize(query.lower())
    query_toks = []
    for w in query_tokens:
        stemmed = stemmer.stem(w)
        if stemmed[-1] == "i":
            query_toks.append(stemmed[: len(stemmed)-1])
        else:
            query_toks.append(stemmed)
    stemmed_query = [" ".join(w for w in query_toks)]

    k_tit = 325
    k_rev = 100

    q_vec_reviews = tfidf_vec_reviews.transform(stemmed_query)
    q_vec_titles = tfidf_vec_titles.transform(stemmed_query)
    q_hat_rev = q_vec_reviews@U_rev[:,:k_rev]
    q_hat_tit = q_vec_titles@U_tit[:,:k_tit]

    sim = []

    for i in range(total_cars):
        num_rev = np.matmul(np.matmul(np.diag(S_rev[:k_rev]),V_T_rev[:k_rev,i]),np.transpose(q_hat_rev))
        denom_rev = np.linalg.norm(np.matmul(np.diag(S_rev[:k_rev]),V_T_rev[:k_rev,i]))*np.linalg.norm(q_hat_rev)
        if denom_rev == 0:
            rev_sc = 0
        else:
            rev_sc = num_rev / denom_rev

        num_tit = np.matmul(np.matmul(np.diag(S_tit[:k_tit]),V_T_tit[:k_tit,i]),np.transpose(q_hat_tit))
        denom_tit = np.linalg.norm(np.matmul(np.diag(S_tit[:k_tit]),V_T_tit[:k_tit,i]))*np.linalg.norm(q_hat_tit)
        if denom_tit == 0:
            tit_sc = 0
        else:
            tit_sc = num_tit / denom_tit
        sim.append((0.3 * rev_sc) + (0.7 * tit_sc))

    return np.array(sim)


    # cos_sim_reviews = cosine_similarity(
    #     tfidf_mat_reviews, q_vec_reviews).flatten()

    # cos_sim_titles = cosine_similarity(
    #     tfidf_mat_titles, q_vec_titles).flatten()
    # cos_sims = (0.3 * cos_sim_reviews) + (0.7 * cos_sim_titles)
    # return cos_sims
