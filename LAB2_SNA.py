import json
import nltk
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D
nltk.download('punkt')
nltk.download('stopwords')
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

if __name__ == "__main__":
    all_wall = []
    file = open(r"wall_asp.txt")
    for line in file.readlines():
        string = line
        wall = json.loads(string)
        json_decode = json.JSONDecoder()
        parsed_response = json_decode.decode(json.dumps(wall))
        nodes = parsed_response.get('items')
        for node in nodes:
            all_wall.append(node.get("text"))

    wordcount = {}
    for wall in all_wall:
        for word in wall.split():
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    try:
        for wc in wordcount:
            print(wc, wordcount[wc])
    except Exception:
        k = 1

    print(str(len(all_wall)) + ' запросов считано')

    stemmer = SnowballStemmer("russian")


    def token_and_stem(text):
        tokens = [word for sent in
                  nltk.sent_tokenize(text) for word in
                  nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[а-яА-Я]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems


    def token_only(text):
        tokens = [word.lower() for sent in
                  nltk.sent_tokenize(text) for word in
                  nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[а-яА-Я]', token):
                filtered_tokens.append(token)
        return filtered_tokens


    # Создаем словари (массивы) из полученных основ
    totalvocab_stem = []
    totalvocab_token = []
    for i in all_wall:
        allwords_stemmed = token_and_stem(i)
        if allwords_stemmed != None:
            totalvocab_stem.extend(allwords_stemmed)
            allwords_tokenized = token_only(i)
            totalvocab_token.extend(allwords_tokenized)

    stopwords = nltk.corpus.stopwords.words('russian')

    # можно расширить список стоп-слов
    stopwords.extend(['что', 'это', 'так', 'вот',
                      'быть', 'как', 'в', 'к', 'на'])

    n_featur = 200000

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000, min_df=0.01, stop_words=stopwords, use_idf=True,
                                       tokenizer=token_and_stem, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_wall)
    print(tfidf_matrix.shape)

    num_clusters = 5

    # Метод к-средних - KMeans
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    idx = km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    print(clusters)
    print(km.labels_)

    # MiniBatchKMeans
    mbk = MiniBatchKMeans(init='random', n_clusters=num_clusters)

    # (init='k-means++',
    # ‘random’ or an ndarray)
    mbk.fit_transform(tfidf_matrix)
    mbk.fit(tfidf_matrix)
    miniclusters = mbk.labels_.tolist()
    print(mbk.labels_)
    # DBSCAN

    db = DBSCAN(eps=0.3, min_samples=10).fit(tfidf_matrix)
    labels = db.labels_
    labels.shape
    print(labels)

    # Аггломеративная класстеризация
    agglo1 = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean')

    # affinity можно выбрать любое или попробовать все по очереди: cosine, l1, l2, manhattan
    answer = agglo1.fit_predict(tfidf_matrix.toarray())
    answer.shape

    # k-means
    clusterkm = km.labels_.tolist()
    # minikmeans
    clustermbk = mbk.labels_.tolist()
    # dbscan
    clusters3 = labels
    # agglo
    # clusters4 = answer.tolist()
    frame = pd.DataFrame(all_wall, index=[clusterkm])
    # k-means
    out = {'title': all_wall, 'cluster': clusterkm}
    frame1 = pd.DataFrame(out, index=[clusterkm],
                          columns=['title', 'cluster'])
    # mini
    out = {'title': all_wall, 'cluster': clustermbk}
    frame_minik = pd.DataFrame(out, index=
    [clustermbk], columns=['title', 'cluster'])
    frame1['cluster'].value_counts()
    frame_minik['cluster'].value_counts()
    dist = 1 - cosine_similarity(tfidf_matrix)
    dist.shape

    # Метод главных компонент - PCA
    icpa = IncrementalPCA(n_components=2, batch_size=16)
    icpa.fit(dist)
    demo2 = icpa.transform(dist)
    xs, ys = demo2[:, 0], demo2[:, 1]

    # PCA 3D
    icpa = IncrementalPCA(n_components=3, batch_size=16)
    icpa.fit(dist)
    ddd = icpa.transform(dist)
    xs, ys, zs = ddd[:, 0], ddd[:, 1], ddd[:, 2]

    # Можно сразу примерно посмотреть, что получится в итоге

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
