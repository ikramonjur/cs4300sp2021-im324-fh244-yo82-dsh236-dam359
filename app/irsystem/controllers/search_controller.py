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

project_name = "Used Car Recommendations"
net_id = "Ikra Monjur: im324, Yoon Jae Oh: yo82, Fareeza Hasan: fh244, Destiny Malloy: dam359, David Hu: dsh236"

mac_memory_in_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)
print("memory beg", mac_memory_in_MB)

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

# absolute_path = os.path.dirname(os.path.abspath(__file__))
# file_path = absolute_path + '/new_reviews.json'

# with open(file_path) as json_file:
#     reviews_dict = json.load(json_file)

# total_cars = len(reviews_dict.keys())

# car_to_id = {car:id for id, car in enumerate(reviews_dict)}
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
# pickle.dump(car_to_id, open("car_to_id.pickle", "wb"))
# pickle.dump(id_to_car, open("id_to_car.pickle", "wb"))
# pickle.dump(total_cars, open("total_cars.pickle", "wb"))

# Load all the pickle files

tfidf_mat_reviews = pickle.load(open("tfidf_mat_reviews.pickle", "rb"))
tfidf_mat_titles = pickle.load(open("tfidf_mat_titles.pickle", "rb"))
ratings = pickle.load(open("ratings.pickle", "rb"))
car_to_id = pickle.load(open("car_to_id.pickle", "rb"))
id_to_car = pickle.load(open("id_to_car.pickle", "rb"))
total_cars = pickle.load(open("total_cars.pickle", "rb"))
tfidf_vec_reviews = pickle.load(open("tfidf_vec_reviews.pickle", "rb"))
tfidf_vec_titles = pickle.load(open("tfidf_vec_titles.pickle", "rb"))

mac_memory_in_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)
print("memory after load", mac_memory_in_MB)

stemmer = SnowballStemmer("english")

def get_ranked(query):
    sim_mat = cosine_sim(query)
    ranked_lst = [(id_to_car[i], ratings[i]) for i in np.argsort(sim_mat)[::-1][:5]]
    # for i in range(len(ranked_lst)):
    #     print(ranked_lst[i])
    return ranked_lst

# data is the car name
def cosine_sim(query):
    query_tokens = treebank_tokenizer.tokenize(query.lower())
    query_toks = []
    for w in query_tokens:
        stemmed = stemmer.stem(w)
        if stemmed[-1] == "i":
            query_toks.append(stemmed[:len(stemmed)-1])
        else:
            query_toks.append(stemmed)
    stemmed_query = [" ".join(w for w in query_toks)]
    q_vec_reviews = tfidf_vec_reviews.transform(stemmed_query)
    cos_sim_reviews = cosine_similarity(tfidf_mat_reviews,q_vec_reviews).flatten()
    q_vec_titles = tfidf_vec_titles.transform(stemmed_query)
    cos_sim_titles = cosine_similarity(tfidf_mat_titles,q_vec_titles).flatten()
    cos_sims = (0.3 * cos_sim_reviews) + (0.7 * cos_sim_titles)
    return cos_sims
