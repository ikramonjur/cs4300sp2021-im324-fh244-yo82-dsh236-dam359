from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import numpy as np
import os

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

absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path = absolute_path + '/reviews.json'

with open(file_path) as json_file:
    reviews_dict = json.load(json_file)

treebank_tokenizer = TreebankWordTokenizer()

# give more weight to the vehicle title if it is in query
def calc_sim_sc(query):
	query_toks = treebank_tokenizer.tokenize(query.lower())
	query_vec = np.array(list(Counter(query_toks).values()))
	sc_dict = {}
	for car in reviews_dict.keys():
		review = reviews_dict[car]['review']
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
		if car == '2002 Dodge Ram Cargo Van 1500 3dr Van (3.9L 6cyl 3A)':
			print("VEC", title_vec)
		title_sc = cosine_sim(query_vec, title_vec)
		review_sc = cosine_sim(query_vec, review_vec)
		# sc_dict[car] = 0.3 * review_sc + 0.7 * title_sc
		sc_dict[car] = title_sc
	return sc_dict

def get_ranked(query):
	sc_dict = calc_sim_sc(query)
	ranked_tup_list = sorted(sc_dict.items(), key=lambda x:x[1], reverse=True)
	print("SCORE:", sc_dict['2002 Dodge Ram Cargo Van 1500 3dr Van (3.9L 6cyl 3A)'])
	ranked_list = []
	for i in range(5):
		ranked_list.append(ranked_tup_list[i][0])
	return ranked_list


# the inputs are already vectors
def cosine_sim(query, data):
	denom = np.linalg.norm(query) * np.linalg.norm(data)
	if denom == 0:
		return 0
	return np.dot(query, data)/denom
