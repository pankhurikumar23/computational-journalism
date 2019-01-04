from gensim import corpora, models
import csv, re
csv.field_size_limit(1000000000)

# Step 1: Reading the csv data into a list of strings called corpus
corpus_unclean = []
with open('state-of-the-union.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for word in reader:
        newStr = re.sub("[?!\-._]", ' ', word[1])
        newRow = [str(int(int(word[0])/10)), newStr.lower().split()]
        corpus_unclean.append(newRow)

# for item in corpus_unclean:
#     print(item[0])

# Step 2: Load the stopwords and remove from corpus
with open('stopwords-en.csv', 'r') as stopwordfile:
    stopwords = stopwordfile.read().split('\n')
# print(stopwords)

corpus_clean1 = [[word for word in document[1] if word not in stopwords] for document in corpus_unclean]
corpus_clean = []
for i in range(len(corpus_unclean)):
    corpus_clean.append([corpus_unclean[i][0], corpus_clean1[i]])

# for i in range(len(corpus_clean)):
#     print(corpus_clean[i][0])

# checking if all stopwords removed
for item in corpus_clean:
    for i in range(len(stopwords)):
        if stopwords[i] in item:
            print("You failed to remove " + stopwords[i] + " from [" + str(item) + "]")

# Step 3: Removing words that occur once
from collections import defaultdict
frequency = defaultdict(int)
for text in corpus_clean:
    for token in text[1]:
        frequency[token] += 1

texts = [[token for token in text[1] if frequency[token] > 1] for text in corpus_clean]

# Step 4: Creating bag-of-words model and generate corpus sparse vector
dictionary = corpora.Dictionary(texts)
# print(type(dictionary.token2id))
corpus = [dictionary.doc2bow(text) for text in texts]
# print(dictionary.token2id)
# print(corpus)

# Step 5: Generating TF-IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
# for document in corpus_tfidf:
#     print(document[1][0])

#TODO: Follow #6 of the assignment now, clarify for LDA/TF-IDF
j = 0
summaryList = []
while j < len(corpus_tfidf):
    if int(corpus_clean[j][0]) < 190 or int(corpus_clean[j][0]) > 200:
        corpus_clean[j][0] = 0
        j += 1
        continue
    if int(corpus_clean[j][0]) >= 190:
        i = j + 1
        decade = int(corpus_clean[j][0])
        first = corpus_tfidf[j]
        print(str(j) + ":" + corpus_clean[j][0])
        print(first)
        while int(corpus_clean[i][0]) == decade:
            adder = corpus_tfidf[i]
            # print(str(i) + ":" + corpus_clean[i][0])
            # print(adder)
            # add the two tf-idf vectors
            a = 0
            b = 0
            while a < len(first) or b < len(adder):
                if (a >= len(first)) or (b >= len(adder)):
                    # if a >= len(first):
                    #     print("a")
                    # else:
                    #     print("b")
                    break
                if (first[a][0] < adder[b][0]):
                    a += 1
                    continue
                if (first[a][0] == adder[b][0]):
                    # print("equal")
                    # print(first[a][0], adder[b][0])
                    # print(first[a][1], adder[b][1])
                    num = first[a][0]
                    sum = first[a][1] + adder[b][1]
                    newList = [i for i in first if i[0] != num]
                    # print(newList)
                    newList.append((num, sum))
                    first = newList
                    # print(first)
                    a += 1
                    b += 1
                    continue
                if (first[a][0] > adder[b][0]):
                    # print("more")
                    # print(first[a][0], adder[b][0])
                    first.append((adder[b][0], adder[b][1]))
                    # print(first)
                    b += 1
            if b < len(adder):
                # print("adding b")
                while b < len(adder):
                    first.append((adder[b][0], adder[b][1]))
                    b += 1
            i += 1
            # print("LAST")
            # print(str(j) + ":" + corpus_clean[j][0])
            # print(first)
        print(first)
        summaryList.append(first)
        # print('\n\n\n')
        j = i

with open("tf-idf.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for item in summaryList:
        writer.writerow(item)

sortedList = []
for item in summaryList:
    sortL = sorted(item, key=lambda t: t[1], reverse=True)[:30]
    sortedList.append(sortL)

print("\n\n\n\n")
count = 1900
dictVect = dictionary.token2id
for item in sortedList:
    print(count)
    count += 10
    for i in range(30):
        if item[i][0] in dictVect.values():
            print(str(item[i][0]), list(dictVect.keys())[list(dictVect.values()).index(item[i][0])], str(item[i][1]))
    print("\n")