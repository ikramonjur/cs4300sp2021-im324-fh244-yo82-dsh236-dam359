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
from bs4 import BeautifulSoup
import requests
import re

project_name = "Used Car Recommendations"
net_id = "Ikra Monjur: im324, Yoon Jae Oh: yo82, Fareeza Hasan: fh244, Destiny Malloy: dam359, David Hu: dsh236"

mac_memory_in_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)
print("memory beg", mac_memory_in_MB)


@irsystem.route('/', methods=['GET'])
def search():
    search_bar = request.args.get('search')
    query = search_bar
    # if manual or automatic add it to query
    transmission = request.args.get('transmission')
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
            data = get_ranked(query)
            # print(data)
        except:
            data = ["No results for current search | Try a new search"]

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


def get_image(car_name):
    car_toks = car_name.split(" ")
    url = 'https://www.google.com/search?q='
    for i, tok in enumerate(car_toks):
        if (i == len(car_toks)-1):
            url += tok+"&tbm=isch"
        else:
            url += tok+'+'
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    car_image = soup.find('img', attrs={'class': 't0fcAb'}).get('src')

    return car_image


def get_link(car_name):
    car_toks = car_name.split(" ")
    url = 'https://www.google.com/search?q='
    for i, tok in enumerate(car_toks):
        if (i == len(car_toks)-1):
            url += tok
        else:
            url += tok+'+'
    url += "&num=" + str(5)
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    # Below code snippet from: https://predictivehacks.com/how-to-scrape-google-results-for-free-using-python/
    result = soup.find_all('div', attrs={'class': 'ZINbbc'})
    results = [re.search(r'\/url\?q\=(.*)\&sa', str(i.find('a', href=True)['href']))
               for i in result if "url" in str(i)]
    links = [i.group(1) for i in results if i != None]
    # end of code snippet

    link = links[0]
    return link


def get_ranked(query):
    query = str(query)
    sim_mat = np.transpose(cosine_sim(query)).flatten()

    if sim_mat[(np.argsort((sim_mat))[::-1][:5])[0]] == 0.0:
        # print("in if")
        return []
    else:
        # print("in else")
        ranked_lst = [(id_to_car[i], round(ratings[i], 2), round(sim_mat[i], 2), get_image(id_to_car[i]), get_link(id_to_car[i]))
                      for i in np.argsort(sim_mat)[::-1][:5]]
        ranked_lst = sorted(
            ranked_lst, key=lambda x: (x[2], x[1]), reverse=True)

        # print("list is ", ranked_lst)

        return ranked_lst
    # printing out to see the cosine scores
    # for i in np.argsort((sim_mat))[::-1][:5]:
        # print(sim_mat[i])


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
    q_hat_rev = q_vec_reviews@U_rev[:, :k_rev]
    q_hat_tit = q_vec_titles@U_tit[:, :k_tit]

    first_mul = np.matmul(
        np.diag(S_rev[:k_rev]), V_T_rev[:k_rev])  # 100 by 15609

    num_rev = np.matmul(np.transpose(first_mul), np.transpose(q_hat_rev))
    # print("second_mul", num_rev.shape) # 15609 by 1

    norms = np.apply_along_axis(np.linalg.norm, 0, first_mul)
    #print("norms", norms.shape)

    norm_q = np.linalg.norm(q_hat_rev)  # this is a float

    denom_rev = norm_q * norms  # 15609
    denom_rev = denom_rev.reshape((denom_rev.shape[0], 1))
    #print("denom rev", denom_rev.shape)

    rev_sc = np.divide(num_rev, denom_rev, out=np.zeros_like(
        num_rev), where=denom_rev != 0)
    #print("rev sc", rev_sc.shape)

    first_num_tit = np.matmul(
        np.diag(S_tit[:k_tit]), V_T_tit[:k_tit])  # 325 by 15609

    num_tit = np.matmul(np.transpose(first_num_tit),
                        np.transpose(q_hat_tit))  # 15609 by 1

    norms_tit = np.apply_along_axis(np.linalg.norm, 0, first_num_tit)

    norm_q_tit = np.linalg.norm(q_hat_tit)

    denom_tit = norm_q_tit * norms_tit
    denom_tit = denom_tit.reshape((denom_tit.shape[0], 1))

    tit_sc = np.divide(num_tit, denom_tit, out=np.zeros_like(
        num_tit), where=denom_tit != 0)

    #print("rev_sc", rev_sc[:5])
    #print("tit_sc", tit_sc[:5])
    sim = (0.3 * rev_sc) + (0.7 * tit_sc)
    return sim


@irsystem.route('/explanation')
def explanation():
    return render_template('explanation.html', name=project_name, netid=net_id)


# def back():
#     back_button = request.args.get('search')
#     return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
