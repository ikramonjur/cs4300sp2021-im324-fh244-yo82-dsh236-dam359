from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, defaultdict
import numpy as np
import os
import math
import pickle
import resource
from sklearn.feature_extraction.text import TfidfVectorizer

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
        data = get_ranked(query)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


# query --> measure sim with "vehicle title + review" --> get the top five similar things

treebank_tokenizer = TreebankWordTokenizer()

absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path = absolute_path + '/reviews.json'

with open(file_path) as json_file:
    reviews_dict = json.load(json_file)

# mapping of car to id - variables are reassigned before computation
total_cars = len(reviews_dict.keys())

car_to_id = {car:id for id, car in enumerate(reviews_dict)}
id_to_car = {id:car for id, car in enumerate(reviews_dict)}

def build_vectorizer(max_n_terms=5000, max_prop_docs=0.95, min_n_docs=1):
    """Returns a TfidfVectorizer object with certain preprocessing properties.

    Params: {max_n_terms: Integer,
             max_prop_docs: Float,
             min_n_docs: Integer}
    Returns: TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(min_df=min_n_docs, max_df=max_prop_docs, max_features=max_n_terms, stop_words='english')
    return vectorizer

tfidf_vec = build_vectorizer()
tfidf_mat_reviews = tfidf_vec.fit_transform([reviews_dict[d]['review'] for d in reviews_dict])
tfidf_mat_titles = tfidf_vec.fit_transform([d for d in reviews_dict])
ratings = [reviews_dict[d]['rating'] for d in reviews_dict]


# def build_tokens_dict():
#     review_tokens = {}
#     for i, car in enumerate(reviews_dict):
#         review_tokens[car] = {"title_toks": treebank_tokenizer.tokenize(car.lower().strip()),
#                               "review_toks": treebank_tokenizer.tokenize(reviews_dict[car][3:].lower().strip())}
#         car_to_id[car] = i
#         id_to_car[i] = car
#     return review_tokens

# def build_inverted_index(review_tokens):
#     title_inverted_index = defaultdict(list)
#     review_inverted_index = defaultdict(list)
#     for car in review_tokens:
#         carid = car_to_id[car]
#         title_toks = review_tokens[car]["title_toks"]
#         review_toks = review_tokens[car]["review_toks"]
#         title_tf = Counter()
#         review_tf = Counter()
#         for tok in title_toks:
#             title_tf[tok] += 1
#         for tok in review_toks:
#             review_tf[tok] += 1
#         for tok in set(title_toks):
#             title_inverted_index[tok].append((carid, title_tf[tok]))
#         for tok in set(review_toks):
#             review_inverted_index[tok].append((carid, review_tf[tok]))
#     return title_inverted_index, review_inverted_index


# def compute_idf(inv_idx, n_docs):
#     idf_dict = {}
#     for term in inv_idx:
#         postings = inv_idx[term]
#         idf = math.log2(n_docs/(1 + len(postings)))
#         idf_dict[term] = idf
#     return idf_dict


# def compute_norms(index, idf, n_docs):
#     norms = np.zeros(n_docs)
#     for term in idf:
#         postings = index[term]
#         for tup in postings:
#             carid = tup[0]
#             norms[carid] += (tup[1] * idf[term]) ** 2
#     norms = np.sqrt(norms)
#     return norms


def index_search(query, index, idf, doc_norms):
    cos_arr = np.zeros(doc_norms.shape[0])
    cos_scores = Counter()
    query_toks = treebank_tokenizer.tokenize(query.lower())
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

# tokens_dict = build_tokens_dict()
# title_inv_idx,  review_inv_idx = build_inverted_index(tokens_dict)
#
# title_idf = compute_idf(title_inv_idx, total_cars)
# title_norms = compute_norms(title_inv_idx, title_idf, total_cars)
#
# review_idf = compute_idf(review_inv_idx, total_cars)
# review_norms = compute_norms(review_inv_idx, review_idf, total_cars)

# Save all of them in pickle files

# pickle.dump(title_inv_idx, open("title_inv_idx.pickle", "wb"))
# pickle.dump(review_inv_idx, open("review_inv_idx.pickle", "wb"))
# pickle.dump(title_idf, open("title_idf.pickle", "wb"))
# pickle.dump(title_norms, open("title_norms.pickle", "wb"))
# pickle.dump(review_idf, open("review_idf.pickle", "wb"))
# pickle.dump(review_norms, open("review_norms.pickle", "wb"))
# pickle.dump(car_to_id, open("car_to_id.pickle", "wb"))
# pickle.dump(id_to_car, open("id_to_car.pickle", "wb"))

# Load all the pickle files

# title_inv_idx = pickle.load(open("title_inv_idx.pickle", "rb"))
# review_inv_idx = pickle.load(open("review_inv_idx.pickle", "rb"))
# title_idf = pickle.load(open("title_idf.pickle", "rb"))
# title_norms = pickle.load(open("title_norms.pickle", "rb"))
# review_idf = pickle.load(open("review_idf.pickle", "rb"))
# review_norms = pickle.load(open("review_norms.pickle", "rb"))
# car_to_id = pickle.load(open("car_to_id.pickle", "rb"))
# id_to_car = pickle.load(open("id_to_car.pickle", "rb"))

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
                        title_sc[carid], float(reviews_dict[car][:3]))
    return sc_dict

# give more weight to the vehicle title if it is in query


def calc_sim_sc(query):
    query_toks = treebank_tokenizer.tokenize(query.lower())
    query_vec = np.array(list(Counter(query_toks).values()))
    sc_dict = {}
    for car in reviews_dict.keys():
        review = reviews_dict[car][3:]
        title_toks = treebank_tokenizer.tokenize(car.lower())
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
                        float(reviews_dict[car][:3]))
    return sc_dict


# def get_ranked(query):
#     sc_dict = calc_sc_inv_idx(query)
#     ranked_tup_list = sorted(sc_dict.items(), key=lambda x: (
#         x[1][0], x[1][1]), reverse=True)
#     ranked_list = []
#     for i in range(5):
#         ranked_list.append((ranked_tup_list[i][0], ranked_tup_list[i][1][1]))
#     return ranked_list

def get_ranked(query):
    sim_mat = calc_sim_tfidf(query)
    ranked_lst = [(id_to_car[i], ratings[i]) for i in np.argsort(sim_mat)[::-1][:5]]
    return ranked_lst

def calc_sim_tfidf(query):
    sim_mat = np.zeros(total_cars)
    for i in range(total_cars):
        sim_mat[i] = (0.3 * cosine_sim(query, id_to_car[i], tfidf_mat_reviews)) + (0.7 * cosine_sim(query, id_to_car[i], tfidf_mat_titles))
        break
    return sim_mat

# data is the car name
def cosine_sim(query, data, tfidf_mat):
    print("in cos sim")
    q_vec = tfidf_vec.transform([query])
    print("query", q_vec.shape)
    car_id = car_to_id[data]
    d_vec = tfidf_mat[car_id]
    print("d", d_vec.shape)
    num = np.dot(q_vec, d_vec)
    denom = np.linalg.norm(q_vec) * np.linalg.norm(d_vec)
    print("out cos sim")
    if denom == 0:
        return 0
    return num/denom
