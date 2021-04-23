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

project_name = "Used Car Recommendations"
net_id = "Ikra Monjur: im324, Yoon Jae Oh: yo82, Fareeza Hasan: fh244, Destiny Malloy: dam359, David Hu: dsh236"


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    if not query:
        data = []
        output_message = ''
    else:
        output_message = "Your search: " + query
        try:
            data = get_ranked(query)
        except:
            data = ["No results for current search | Try a new search"]
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


# query --> measure sim with "vehicle title + review" --> get the top five similar things

treebank_tokenizer = TreebankWordTokenizer()
#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path = absolute_path + '/smaller_reviews.json'

with open(file_path) as json_file:
    reviews_dict = json.load(json_file)

# mapping of car to id - variables are reassigned before computation
car_to_id = {}
id_to_car = {}

total_cars = len(reviews_dict.keys())


def build_tokens_dict():
    review_tokens = {}
    for i, car in enumerate(reviews_dict):
        # review_tokens[car] = {"title_toks": treebank_tokenizer.tokenize(car.lower().strip()),
        # "review_toks": treebank_tokenizer.tokenize(reviews_dict[car]["review"].lower().strip())}
        review_tokens[car] = {"title_toks": treebank_tokenizer.tokenize(car.lower().strip()),
                              "review_toks": treebank_tokenizer.tokenize(reviews_dict[car]['review'].lower().strip())}
        car_to_id[car] = i
        id_to_car[i] = car
    return review_tokens


def build_inverted_index(review_tokens):
    title_inverted_index = defaultdict(list)
    review_inverted_index = defaultdict(list)
    for car in review_tokens:
        carid = car_to_id[car]
        title_tokens = review_tokens[car]["title_toks"]
        title_toks = []
        for w in title_tokens:
            # stem the title tokens
            stemmed = stemmer.stem(w)
            if w.isalnum():
                if stemmed[-1] == "i":
                    title_toks.append(stemmed[:len(stemmed)-1])
                else:
                    title_toks.append(stemmed)
                # Map A-->automatic, M--> manual, Map AM -> automatic
                if len(w) == 2 and w[0].isdigit() and w[1].isalpha():
                    if w[1] == "a":
                        title_toks.append(stemmer.stem("automatic"))
                    elif w[1] == "m":
                        title_toks.append(stemmer.stem("manual"))
                elif w[0].isdigit() and w.endswith('am'):
                    title_toks.append(stemmer.stem("automatic"))
        review_toks = review_tokens[car]["review_toks"]
        title_tf = Counter()
        review_tf = Counter()
        for tok in title_toks:
            title_tf[tok] += 1
        for tok in review_toks:
            review_tf[tok] += 1
        for tok in set(title_toks):
            title_inverted_index[tok].append((carid, title_tf[tok]))
        for tok in set(review_toks):
            review_inverted_index[tok].append((carid, review_tf[tok]))
    return title_inverted_index, review_inverted_index


def compute_idf(inv_idx, n_docs):
    idf_dict = {}
    for term in inv_idx:
        postings = inv_idx[term]
        idf = math.log2(n_docs/(1 + len(postings)))
        idf_dict[term] = idf
    return idf_dict


def compute_norms(index, idf, n_docs):
    norms = np.zeros(n_docs)
    for term in idf:
        postings = index[term]
        for tup in postings:
            carid = tup[0]
            norms[carid] += (tup[1] * idf[term]) ** 2
    norms = np.sqrt(norms)
    return norms


def index_search(query, index, idf, doc_norms):
    cos_arr = np.zeros(doc_norms.shape[0])
    cos_scores = Counter()
    query_tokens = treebank_tokenizer.tokenize(query.lower())
    query_toks = []
    for w in query_tokens:
        stemmed = stemmer.stem(w)
        if stemmed[-1] == "i":
            query_toks.append(stemmed[:len(stemmed)-1])
        else:
            query_toks.append(stemmed)
    query_norm = 0
    q_tf = Counter(query_toks)

    # looping through each token in query
    for token in q_tf.keys():
        if token in idf.keys():  # make sure token appears in at least one doc
            query_norm += (q_tf[token] * idf[token]) ** 2
            postings = index[token]
            for tup in postings:
                carid = tup[0]
                doc_tf = tup[1]
                cos_arr[carid] += q_tf[token] * \
                    idf[token] * idf[token] * doc_tf

    query_norm = math.sqrt(query_norm)
    for i in range(doc_norms.shape[0]):
        if cos_arr[i] != 0:
            cos_sc = cos_arr[i] / (query_norm * doc_norms[i])
            cos_scores[i] = cos_sc
    return cos_scores

# Setup computation for similarity score calculations


tokens_dict = build_tokens_dict()
title_inv_idx,  review_inv_idx = build_inverted_index(tokens_dict)

title_idf = compute_idf(title_inv_idx, total_cars)
title_norms = compute_norms(title_inv_idx, title_idf, total_cars)

review_idf = compute_idf(review_inv_idx, total_cars)
review_norms = compute_norms(review_inv_idx, review_idf, total_cars)

""" # Save all of them in pickle files

pickle.dump(title_inv_idx, open("title_inv_idx.pickle", "wb"))
pickle.dump(review_inv_idx, open("review_inv_idx.pickle", "wb"))
pickle.dump(title_idf, open("title_idf.pickle", "wb"))
pickle.dump(title_norms, open("title_norms.pickle", "wb"))
pickle.dump(review_idf, open("review_idf.pickle", "wb"))
pickle.dump(review_norms, open("review_norms.pickle", "wb"))
pickle.dump(car_to_id, open("car_to_id.pickle", "wb"))
pickle.dump(id_to_car, open("id_to_car.pickle", "wb")) """

# Load all the pickle files
'''
title_inv_idx = pickle.load(open("title_inv_idx.pickle", "rb"))
review_inv_idx = pickle.load(open("review_inv_idx.pickle", "rb"))
title_idf = pickle.load(open("title_idf.pickle", "rb"))
title_norms = pickle.load(open("title_norms.pickle", "rb"))
review_idf = pickle.load(open("review_idf.pickle", "rb"))
review_norms = pickle.load(open("review_norms.pickle", "rb"))
car_to_id = pickle.load(open("car_to_id.pickle", "rb"))
id_to_car = pickle.load(open("id_to_car.pickle", "rb"))
'''


def calc_sc_inv_idx(query):
    #tokens_dict = build_tokens_dict()
    #title_inv_idx,  review_inv_idx = build_inverted_index(tokens_dict)

    # calculating title scores
    title_sc = index_search(query, title_inv_idx, title_idf, title_norms)

    # calculating review scores
    review_sc = index_search(query, review_inv_idx, review_idf, review_norms)

    # print("TITLE SCORE", title_sc)
    sc_dict = {}
    cars_with_sc = set(title_sc.keys()).union(set(review_sc.keys()))
    for carid in cars_with_sc:
        car = id_to_car[carid]
        sc_dict[car] = (0.3 * review_sc[carid] + 0.7 *
                        title_sc[carid], float(reviews_dict[car]["rating"]))
    return sc_dict

# give more weight to the vehicle title if it is in query


def calc_sim_sc(query):
    # tokenize the query
    query_tokens = treebank_tokenizer.tokenize(query.lower())

    # stem the query
    query_toks = []
    for w in query_tokens:
        query_toks.append(stemmer.stem(w))

    query_vec = np.array(list(Counter(query_toks).values()))
    sc_dict = {}
    for car in reviews_dict.keys():
        review = reviews_dict[car]["review"]
        title_tokens = treebank_tokenizer.tokenize(car.lower())
        title_toks = []
        for w in title_tokens:
            title_toks.append(stemmer.stem(w))
        title_qtf = Counter()
        review_toks = treebank_tokenizer.tokenize(review.lower())
        review_qtf = Counter()
        for token in query_toks:
            if token in title_toks:
                title_qtf[token] += 1
            else:
                title_qtf[token] = 0
            if token in review_toks:
                review_qtf[token] += 1
            else:
                review_qtf[token] = 0
        review_vec = np.array(list(review_qtf.values()))
        title_vec = np.array(list(title_qtf.values()))
        title_sc = cosine_sim(query_vec, title_vec)
        review_sc = cosine_sim(query_vec, review_vec)
        sc_dict[car] = (0.3 * review_sc + 0.7 * title_sc,
                        float(reviews_dict[car]["rating"]))
    return sc_dict


def get_ranked(query):
    sc_dict = calc_sc_inv_idx(query)
    ranked_tup_list = sorted(sc_dict.items(), key=lambda x: (
        x[1][0], x[1][1]), reverse=True)
    ranked_list = []
    for i in range(5):
        ranked_list.append((ranked_tup_list[i][0], ranked_tup_list[i][1][1]))
    return ranked_list


# the inputs are already vectors
def cosine_sim(query, data):
    denom = np.linalg.norm(query) * np.linalg.norm(data)
    if denom == 0:
        return 0
    return np.dot(query, data)/denom
