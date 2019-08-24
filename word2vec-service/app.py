"""Cloud Foundry test"""
from flask import Flask
import cf_deployment_tracker
import os

from datetime import datetime
from flask import Flask, jsonify, redirect, render_template, request, url_for, make_response
from gensim.models import word2vec, KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys
import json

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpld3
from mpld3 import plugins

# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

# On Bluemix, get the port number from the environment variable VCAP_APP_PORT
# When running this app on the local machine, default the port to 8080
port = int(os.getenv('VCAP_APP_PORT', 8080))

# model = word2vec.Word2Vec.load_word2vec_format('models/text8.bin', binary=True)
# model = word2vec.Word2Vec.load_word2vec_format('models/animedb.bin', binary=True)
model = KeyedVectors.load_word2vec_format('models/text8.bin', binary=True)
# model = word2vec.Word2Vec.load_word2vec_format('models/jph2016032.bin', binary=True)
model.init_sims(replace=True)

def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in model.vocab]


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/search')
def search():
    """Renders the search page."""
    return render_template(
        'search.html',
        title='Find the top-N most similar words',
        year=datetime.now().year,
        message='search page.'
    )

@app.route('/restapitester')
def restapitester():
    """Renders the tester page."""
    return render_template(
        'restapitester.html',
        title='Rest API Tester',
        year=datetime.now().year,
        message='rest api tester page.'
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='application description page.'
    )

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")

@app.route('/most_similar', methods=['GET', 'POST'], strict_slashes=False)
def most_similar():
    if request.method == 'POST':
        # keyword = request.form['keyword']
        positives = [item for item in request.form.getlist('positive[]') if item]
        negatives = [item for item in request.form.getlist('negative[]') if item]
        n = request.form['topn']
        clustering_name = request.form['clustering_name']
        nc = request.form['n_clusters']
                
    else:
        # keyword = request.args.get('word')
        positives = [item for item in request.args.getlist('positive[]') if item]
        negatives = [item for item in request.args.getlist('negative[]') if item]
        n = request.args.get('topn')
        clustering_name = request.args.get('clustering_name')
        nc = request.args.get('n_clusters')
                        
    if not n:
        topn = 10
    else:
        topn = int(n)
        
    if not nc:
        n_clusters = 3
    else:
        n_clusters = int(nc)

    try:
        # most_similar = model.most_similar(positive=positives, negative=negatives, topn=topn)
        most_similar = get_most_similar(positives, negatives, topn)
        fig_to_html = html_tooltips(positives, negatives, most_similar)
    except:
        print(sys.exc_info()[0])
        most_similar = []
        fig_to_html = ""
    
    if request.method == 'POST':
        return render_template(
            'result.html',
            title='Result',
            year=datetime.now().year,
            positives=positives,
            negatives=negatives,
            topn=topn,
            clustering_name=clustering_name,
            n_clusters=n_clusters,
            most_similar=json.dumps(most_similar, indent=4, ensure_ascii=False),
            fig_to_html=fig_to_html
        )
    else:
        return jsonify(most_similar)

@app.route('/pca_most_similar', methods=['GET', 'POST'], strict_slashes=False)
def pca_most_similar():
    if request.method == 'POST':
        # keyword = request.form['keyword']
        positives = [item for item in request.form.getlist('positive[]') if item]
        negatives = [item for item in request.form.getlist('negative[]') if item]
        n = request.form['topn']
        clustering_name = request.form['clustering_name']
        nc = request.form['n_clusters']
        
    else:
        # keyword = request.args.get('word')
        positives = [item for item in request.args.getlist('positive[]') if item]
        negatives = [item for item in request.args.getlist('negative[]') if item]
        n = request.args.get('topn')
        clustering_name = request.args.get('clustering_name')
        nc = request.args.get('n_clusters')
        
    if not n:
        topn = 10
    else:
        topn = int(n)

    if not nc:
        n_clusters = 3
    else:
        n_clusters = int(nc)
        
    try:
        # most_similar = model.most_similar(positive=positives, negative=negatives, topn=topn)
        most_similar = get_most_similar(positives, negatives, topn)
    except:
        # most_similar=sys.exc_info()[0]
        most_similar = []
        return jsonify(most_similar)

    # print(get_pca_cluster_comparison(positives, negatives, most_similar, algorithm_name, n_clusters))
    # print(get_pca_cluster_comparison(positives, negatives, most_similar, 'KMeans', 3))
        
    # return get_pca_transform(positives, negatives, most_similar)
    return get_pca_cluster_comparison(positives, negatives, most_similar, clustering_name, n_clusters)

def get_most_similar(positives, negatives, topn):
    try:
        most_similar = model.most_similar(positive=positives, negative=negatives, topn=topn)
    except:
        print(sys.exc_info()[0])
        most_similar = []

    return most_similar

def get_x_pca_transform(positives, negatives, most_similar):
    # # print(model.vector_size)
    # arr = model[keyword]
    # keys = [keyword]
    arr = np.zeros((0,model.vector_size))
    keys = []
    
    # for key in list(set(positives + negatives)):
    for key in (positives + negatives):
        keys.append(key)
        arr = np.vstack((arr, model[key]))

    for i in range(len(most_similar)):
        # print(most_similar[i][0])
        key = most_similar[i][0]
        keys.append(key)
        arr = np.vstack((arr, model[key]))
    
    # print(arr.shape)
    # print(arr)
    
    # 主成分分析による次元削減
    pca = PCA(n_components=2)
    pca.fit(arr)

    # print(pca.explained_variance_ratio_)
    
    x_pca = pca.transform(arr)
    
    # 主成分分析後のサイズ
    # print(x_pca.shape)
    # print(x_pca)

    return x_pca, keys

# def get_pca_transform(keyword, most_similar):
def get_pca_transform(positives, negatives, most_similar):
    
    x_pca, keys = get_x_pca_transform(positives, negatives, most_similar)

    N = len(keys)
    # print(N)

    features = x_pca
    # K-means クラスタリングをおこなう
    # この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
    # kmeans_model = KMeans(n_clusters=3, random_state=10).fit(features)
    kmeans_model = KMeans(n_clusters=3).fit(features)
    
    # 分類先となったラベルを取得する
    labels = kmeans_model.labels_
    
    # # ラベルを表示する
    # for label, feature in zip(labels, features):
    #     print(label, feature)
    
    cluster_centers = kmeans_model.cluster_centers_
    # print(cluster_centers)
    # print([{"x": str(cluster_center[0]), "y": str(cluster_center[1])} for cluster_center in cluster_centers])
    # print([{"_id": i, "x": str(cluster_center[0]), "y": str(cluster_center[1])} for i, cluster_center in enumerate(cluster_centers)])
    centers = [{"_id": i, "x": str(cluster_center[0]), "y": str(cluster_center[1])} for i, cluster_center in enumerate(cluster_centers)]
    nodes = [{"_id": i, "key": str(keys[i]), "x": str(x_pca[i][0]), "y": str(x_pca[i][1]), "group": str(labels[i])} for i in range(N)]
    
    return jsonify([{"cluster_centers": centers, "cluster_nodes": nodes}])           
    # return jsonify([{"_id": i, "key": str(keys[i]), "x": str(x_pca[i][0]), "y": str(x_pca[i][1]), "group": str(labels[i])} for i in range(N)])
    # return jsonify([{"_id": i, "key": str(keys[i]), "x": str(x_pca[i][0]), "y": str(x_pca[i][1])} for i in range(N)])
    # return x_pca

def get_pca_cluster_comparison(positives, negatives, most_similar, clustering_name, n_clusters):
    import time
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    import math
        
    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(0)
    
    x_pca, keys = get_x_pca_transform(positives, negatives, most_similar)

    N = len(keys)
    
    w2v = x_pca, None
    
    clustering_names = [ 'KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering',
                         'Ward', 'AgglomerativeClustering', 'DBSCAN', 'Birch' ]
    
    plot_num = 1

    # clusters = []
    clusters = {}
        
    datasets = [w2v]
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        # X = StandardScaler().fit_transform(X)
    
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
    
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
        affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)    
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=n_clusters, connectivity=connectivity)
        dbscan = cluster.DBSCAN(eps=.2)    
        birch = cluster.Birch(n_clusters=n_clusters)

        clustering_algorithms = [ kmeans, two_means, affinity_propagation, ms, spectral,
                                  ward, average_linkage, dbscan, birch ]
                
        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            algorithm.fit(X)
            # if hasattr(algorithm, 'labels_'):
            #     labels = algorithm.labels_
            if hasattr(algorithm, 'labels_'):
                labels = algorithm.labels_.astype(np.int)
            else:
                labels = algorithm.predict(X)
                
            if hasattr(algorithm, 'cluster_centers_'):
                cluster_centers = algorithm.cluster_centers_
            
            centers = [{"_id": i, "x": str(cluster_center[0]), "y": str(cluster_center[1])} for i, cluster_center in enumerate(cluster_centers)]
            nodes = [{"_id": i, "key": str(keys[i]), "x": str(x_pca[i][0]), "y": str(x_pca[i][1]), "group": str(labels[i])} for i in range(N)]

            # clusters.append({name : [{"cluster_centers": centers, "cluster_nodes": nodes}]})
            clusters[name] = [{"cluster_centers": centers, "cluster_nodes": nodes}]

    # print(clusters[clustering_name])
    
    return jsonify(clusters[clustering_name])
    # return jsonify(clusters)
    # return clusters

@app.route('/plot_cluster_comparison', methods=['GET'], strict_slashes=False)
def plot_cluster_comparison():
    if request.method == 'POST':
        # keyword = request.form['keyword']
        positives = [item for item in request.form.getlist('positive[]') if item]
        negatives = [item for item in request.form.getlist('negative[]') if item]
        n = request.form['topn']
        nc = request.form['n_clusters']
        
    else:
        # keyword = request.args.get('word')
        positives = [item for item in request.args.getlist('positive[]') if item]
        negatives = [item for item in request.args.getlist('negative[]') if item]
        n = request.args.get('topn')
        nc = request.args.get('n_clusters')

    if not n:
        topn = 10
    else:
        topn = int(n)

    if not nc:
        n_clusters = 3
    else:
        n_clusters = int(nc)

    try:
        # most_similar = model.most_similar(positive=positives, negative=negatives, topn=topn)
        most_similar = get_most_similar(positives, negatives, topn)
    except:
        # most_similar=sys.exc_info()[0]
        most_similar = []
        return jsonify(most_similar)
    
    return get_plot_cluster_comparison(positives, negatives, most_similar, n_clusters)

# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def get_plot_cluster_comparison(positives, negatives, most_similar, n_clusters):
    import time
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    import math
        
    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(0)
    
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # n_samples = 1500
    # noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
    #                                       noise=.05)
    # noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    # blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    # no_structure = np.random.rand(n_samples, 2), None

    x_pca, keys = get_x_pca_transform(positives, negatives, most_similar)

    w2v = x_pca, None
    
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    clustering_names = [ 'KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering',
                         'Ward', 'AgglomerativeClustering', 'DBSCAN', 'Birch' ]
    
    # plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
    #                     hspace=.01)
    fig = plt.figure(figsize=(11, 11))
    plt.subplots_adjust(left=.02, right=.98, bottom=.02, top=.96, wspace=.05, hspace=.15)
            
    plot_num = 1
            
    # datasets = [noisy_circles, noisy_moons, blobs, no_structure]
    datasets = [w2v]
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        # X = StandardScaler().fit_transform(X)
    
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
    
        # create clustering estimators
        # two_means = cluster.MiniBatchKMeans(n_clusters=2)
        # affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)    
        # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # spectral = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors")
        # ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity)
        # average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=2, connectivity=connectivity)
        # dbscan = cluster.DBSCAN(eps=.2)    
        # birch = cluster.Birch(n_clusters=2)

        kmeans = cluster.KMeans(n_clusters=n_clusters)
        two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
        affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)    
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=n_clusters, connectivity=connectivity)
        dbscan = cluster.DBSCAN(eps=.2)    
        birch = cluster.Birch(n_clusters=n_clusters)

        clustering_algorithms = [ kmeans, two_means, affinity_propagation, ms, spectral,
                                  ward, average_linkage, dbscan, birch ]

        cols = 3
        rows = math.ceil(len(clustering_algorithms) / cols)
        
        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
    
            # plot
            # plt.subplot(4, len(clustering_algorithms), plot_num)
            plt.subplot(rows, cols, plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    
            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
    
    # plt.show()
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()

    response = make_response(data)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Length'] = len(data)
    return response

# http://mpld3.github.io/examples/html_tooltips.html
# def html_tooltips(keyword, most_similar):
def html_tooltips(positives, negatives, most_similar):
    
    x_pca, keys = get_x_pca_transform(positives, negatives, most_similar)
    
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
        
    # N = 50
    # df = pd.DataFrame(index=range(N))
    # df['x'] = np.random.randn(N)
    # df['y'] = np.random.randn(N)
    # df['z'] = np.random.randn(N)
    N = len(keys)
    # print(N)
    df = pd.DataFrame(index=range(N))
    df['x'] = x_pca[:, 0]
    df['y'] = x_pca[:, 1]

    # print(x_pca[:, 0])
    # print(x_pca[:, 1])
    
    labels = []
    for i in range(N):
        # label = df.ix[[i], :].T
        # label.columns = ['Row {0}'.format(i)]
        label = df.ix[[i], :].T
        # print(keys[i])
        label.columns = ['Word: {0:s}'.format(str(keys[i]))]
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html()))
            
    points = ax.plot(df.x, df.y, 'o', color='b', mec='k', ms=15, mew=1, alpha=.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_title('PCA for top-N most similar words: ' + keyword, size=20)
    ax.set_title('PCA for top-N most similar words', size=20)

    # tooltip = plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=css)
    tooltip = plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)
    
    # mpld3.show()
    return mpld3.fig_to_html(fig)


@app.route('/similarity', methods=['GET', 'POST'], strict_slashes=False)
def similarity():
    if request.method == 'POST':
        word1 = request.form['word1']
        word2 = request.form['word2']
    else:
        word1 = request.args.get('word1')
        word2 = request.args.get('word2')
    
    try:
        similarity=model.similarity(word1, word2)
    except:
        similarity=sys.exc_info()[0]
    
    result = {
        "word1": word1,
        "word2": word2,
        "similarity": similarity,
    }

    return jsonify(result)


@app.route('/doesnt_match/<words>', methods=['GET'], strict_slashes=False)
def doesnt_match(words):
    word_list = words.split("+")
    return jsonify({"doesnt_match": model.doesnt_match(word_list), "word_list": word_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
