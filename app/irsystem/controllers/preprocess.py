from csv import reader
import json
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

stemmer = PorterStemmer()
treebank_tokenizer = TreebankWordTokenizer()

absolute_path = os.path.dirname(os.path.abspath(__file__))
directory = absolute_path + '/folder' #this folder contains all the csv files which is not pushed to git

def parse_csv():
    """
    this function parses the csv data files and gets the preprocessed necessary information.
    all the stemming of the words and the mapping of transmissions in the title occurs in here.
    returns a dictionary that contains the stemmed reviews, stemmed title, and average rating for each car
    """
    data = {} # saves the processed data from csv to dump to json
    for filename in os.listdir(directory):
        with open('folder/'+filename, 'r') as read_obj:
            print(filename)
            csv_reader = reader(read_obj)
            last_key = ""
            row_counter = 0
            for i, row in enumerate(csv_reader):
                if (i==0): # for the header row - do nothing
                    continue
                if (row[0] == str(row_counter)):
                    if (row[3] in data.keys()):
                        stripped_data = (row[4] + row[5]).strip('\n')
                        data[row[3]]["review"] += [" ".join(stemmer.stem(word.strip('\n')) for word in stripped_data .split(" "))][0]
                        if len(row) == 7:
                            data[row[3]]["rating"] += float(row[6])
                        data[row[3]]["review_count"] += 1
                    else:
                        stripped_data = (row[4] + row[5]).strip('\n')
                        title_tokens = treebank_tokenizer.tokenize(row[3].lower().strip())
                        title_toks = []
                        for w in title_tokens:
                            stemmed = stemmer.stem(w)
                            if w.isalnum():
                                if w[-1] == "i":
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
                        data[row[3]] = {"title": [" ".join(word for word in title_toks)][0], "review" : [" ".join(stemmer.stem(word.strip('\n')) for word in stripped_data.split(" "))][0], "rating" : float(row[6]) if len(row) == 7 else 0.0, "review_count" : 1}
                    last_key = row[3]
                    row_counter += 1
                else:
                    stripped_data = (row[0]).strip('\n')
    #                 print(stripped_data)
                    data[last_key]["review"] += [" ".join(stemmer.stem(word.strip('\n')) for word in stripped_data.split(" "))][0]
                    if (len(row) == 2):
                        data[last_key]["rating"] += float(row[1])
    
    # getting the average rating
    for d in data.keys():
        data[d]["rating"] /= data[d]["review_count"]

    # getting only cars that have more than 3 reviews
    new_data = {}
    for d in data:
        if data[d]["review_count"] > 3:
            new_review = data[d]["review"].replace("\n", " ")
            new_data[d] = {"title": data[d]["title"], "review": new_review, "rating": data[d]["rating"]}
    return new_data

# create json file
car_data = parse_csv()
with open("car_reviews.json", "w") as outfile: 
    json.dump(car_data, outfile)



