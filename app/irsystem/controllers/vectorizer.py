import pickle
import os
import json
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer


absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path = absolute_path + '/car_reviews.json' # this is the reviews json file created after preprocessing, not pushed to github

with open(file_path) as json_file:
        reviews_dict = json.load(json_file)

total_cars = len(reviews_dict.keys())

id_to_car = {id:car for id, car in enumerate(reviews_dict)}

ratings = [reviews_dict[d]['rating'] for d in reviews_dict]

def vectorize(option):
    """
    this function creates the tfidf vectorizer for the reviews or titles of our cars in our json dataset
    given the option parameter (can be either "title" or "review")
    """
    tfidf_vec = TfidfVectorizer()
    tfidf_mat = tfidf_vec.fit_transform([reviews_dict[d][option] for d in reviews_dict])
    return tfidf_vec, tfidf_mat

# Save all of them in pickle files
def pickle_dump(): 
    path_to_app = "../../../" #using this so the files are created in the outermost directory
    pickle.dump(ratings, open(path_to_app + "ratings.pickle", "wb"))
    pickle.dump(id_to_car, open(path_to_app + "id_to_car.pickle", "wb"))
    pickle.dump(total_cars, open(path_to_app + "total_cars.pickle", "wb"))

    # save the tfidf in pickles
    pickle.dump(tfidf_vec_reviews, open(path_to_app + "tfidf_vec_reviews.pickle", "wb"))
    pickle.dump(tfidf_mat_reviews, open(path_to_app + "tfidf_mat_reviews.pickle", "wb"))
    pickle.dump(tfidf_vec_titles, open(path_to_app + "tfidf_vec_titles.pickle", "wb"))
    pickle.dump(tfidf_mat_titles, open(path_to_app + "tfidf_mat_titles.pickle", "wb"))

    #save the svd in pickles
    pickle.dump(U_tit, open(path_to_app + "u_tit_svd.pickle", "wb"))
    pickle.dump(S_tit, open(path_to_app + "s_tit_svd.pickle", "wb"))
    pickle.dump(V_T_tit, open(path_to_app + "v_t_svd.pickle", "wb"))
    pickle.dump(U_rev, open(path_to_app + "u_rev_svd.pickle", "wb"))
    pickle.dump(S_rev, open(path_to_app + "s_rev_svd.pickle", "wb"))
    pickle.dump(V_T_rev, open(path_to_app + "v_t_rev_svd.pickle", "wb"))

tfidf_vec_reviews, tfidf_mat_reviews = vectorize('review')
tfidf_vec_titles, tfidf_mat_titles = vectorize('title')
U_tit, S_tit, V_T_tit = svds(tfidf_mat_titles.T, k=10)
U_rev, S_rev, V_T_rev = svds(tfidf_mat_reviews.T, k=10)
pickle_dump()
    

