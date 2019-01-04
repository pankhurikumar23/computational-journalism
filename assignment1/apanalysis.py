from gensim import corpora, models
import csv, re

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Step 1: Reading the csv data into a list of strings called corpus
corpus_unclean = []
with open('ap.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for word in reader:
        newStr = re.sub("[?!\-._]", ' ', word[1])
        newRow = [word[0], newStr.lower().split()]
        corpus_unclean.append(newRow)

# for item in corpus_unclean:
#     print(item)

# Step 2: Load the stopwords and remove from corpus
with open('stopwords-en.csv', 'r') as stopwordfile:
    stopwords = stopwordfile.read().split('\n')
# print(stopwords)

corpus_clean = [[word for word in document[1] if word not in stopwords] for document in corpus_unclean]

# checking if all stopwords removed
for item in corpus_clean:
    for i in range(len(stopwords)):
        if stopwords[i] in item:
            print("You failed to remove " + stopwords[i] + " from [" + str(item) + "]")

# Step 3: Removing words that occur once
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in corpus_clean:
#     for token in text:
#         frequency[token] += 1
#
# texts = [[token for token in text if frequency[token] > 1] for text in corpus_clean]
texts = corpus_clean

# Step 4: Creating bag-of-words model and generate corpus sparse vector
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# print(dictionary.token2id)

# Step 5: Generating TF-IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
# for document in corpus_tfidf:
#     print(document)

#TODO: learn how to read this and annotate the topic and observe the stuff
num_topic = [100, 200]
#### LSI ##########
for num_topics in num_topic:
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics, power_iters=300)
    corpus_lsi = lsi_model[corpus_tfidf]

    #writing doc classification to file
    count = 0
    filename = 'lsi_list_' + str(num_topics) + '.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for document in corpus_lsi:
            count += 1
            try:
                writer.writerow(max(document, key = lambda item:item[1]))
            except ValueError:
                print(count)

    name_of_file = 'lsi_topics_' + str(num_topics) + '.csv'
    with open(name_of_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\n')
        writer.writerow(lsi_model.print_topics(num_topics))

    #### LDA #######
    lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)#, iterations = 300, passes = 5)
    corpus_lda = lda_model[corpus_tfidf]

    #writing doc classification to file
    count = 0
    filename = 'lda_list_' + str(num_topics) + '.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for document in corpus_lda:
            count += 1
            try:
                writer.writerow(max(document, key = lambda item:item[1]))
            except ValueError:
                print(count)

    name_of_file = 'lda_topics_' + str(num_topics) + '.csv'
    with open(name_of_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\n')
        writer.writerow(lda_model.print_topics(num_topics))