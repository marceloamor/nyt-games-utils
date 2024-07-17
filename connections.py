""" -CONNECTIONS SOLVER-
current plan:
- create semantic distance matrix through pairwise comparison of words]
- use kmeans clustering to group words by semantic similarity
- if groups are uneven, recluster to ensure 4 groups of 4 words each, minimising within-cluster semantic distance (wcss)


notes:
use following link: https://www.geeksforgeeks.org/python-word-similarity-using-spacy/
- get semantically similar word rankings to segregate words into groups
- maybe a cheeky llm api call?

# Downloading the small model containing tensors.
python -m spacy download en_core_web_sm

# Downloading over 1 million word vectors.
python -m spacy download en_core_web_lg

"""

import spacy
import numpy as np
from sklearn.cluster import KMeans


nlp = spacy.load('en_core_web_md') 
  
ydays_words = [
    "lady",
    "perfect",
    "goodness",
    "maple",
    "future",
    "mercy",
    "drummer",
    "simple",
    "ring",
    "cough",
    "heavens",
    "swan",
    "present",
    "lord",
    "corn",
    "past",
]

todays_words2 = [
    "aces",
    "globe",
    "head",
    "grow",
    "bubble",
    "swell",
    "neato",
    "lather",
    "nifty",
    "marble",
    "foam",
    "mount",
    "froth",
    "build",
    "pearl",
    "keen",
]

todays_words = [
    "mini",
    "mouse",
    "alone",
    "knock",
    "ram",
    "lily",
    "slam",
    "bachelor",
    "pan",
    "chopped",
    "jaguar",
    "catfish",
    "maxi",
    "survivor",
    "roast",
    "fiat",
]


def calc_semantic_distance(word1, word2):
    words = " ".join([word1, word2]) 
    
    tokens = nlp(words) 
    
    for token in tokens: 
        # Printing the following attributes of each token. 
        # text: the word string
        # has_vector: if it contains a vector representation in the model
        # vector_norm: the algebraic norm of the vector
        # is_oov: if the word is out of vocabulary
        print(token.text, token.has_vector, token.vector_norm, token.is_oov) 
    
    token1, token2 = tokens[0], tokens[1] 
    
    print("Similarity:", token1.similarity(token2)) 
    return token1.similarity(token2)

def calc_distance_matrix(todays_words):
    distance_matrix = np.zeros((len(todays_words), len(todays_words)))

    # calc semantic differences for each pair of words
    for i in range(len(todays_words)):
        for j in range(i+1, len(todays_words)):
            semantic_distance = calc_semantic_distance(todays_words[i], todays_words[j])
            distance_matrix[i][j] = semantic_distance
            distance_matrix[j][i] = semantic_distance
    print(distance_matrix)
    return distance_matrix

def calc_clusters_kmeans(words, distance_matrix):
    num_clusters = 4

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(distance_matrix)

    # Get the cluster labels for each word
    cluster_labels = kmeans.labels_

    # Group the words based on cluster labels
    word_groups = {}
    for i in range(num_clusters):
        word_groups[i] = [words[j] for j, label in enumerate(cluster_labels) if label == i]
    
    print(word_groups)
    return word_groups


def adjust_clusters(initial_clusters, distance_matrix):
    # Calculate the initial WCSS
    kmeans = KMeans(n_clusters=len(initial_clusters))
    kmeans.fit(distance_matrix)
    initial_wcss = kmeans.inertia_

    # Define a function to calculate the WCSS for a set of clusters
    def calculate_wcss(clusters):
        wcss = 0
        for cluster in clusters:
            cluster_center = np.mean([calc_semantic_distance(word1, word2) for word1 in cluster for word2 in cluster])
            wcss += sum([(calc_semantic_distance(word, cluster_center) ** 2) for word in cluster])
        return wcss

    # Continue swapping words to minimize the WCSS
    adjusted_clusters = initial_clusters.copy()
    while True:
        # Calculate the current WCSS
        current_wcss = calculate_wcss(adjusted_clusters)

        # Flag to track if any swaps were made
        swapped = False

        for i in range(len(initial_clusters)):
            for j in range(len(initial_clusters)):
                if len(adjusted_clusters[i]) > 4 and len(adjusted_clusters[j]) < 4:
                    for word in adjusted_clusters[i]:
                        new_cluster_j = adjusted_clusters[j] + [word]
                        new_cluster_i = [w for w in adjusted_clusters[i] if w != word]
                        new_clusters = adjusted_clusters.copy()
                        new_clusters[i] = new_cluster_i
                        new_clusters[j] = new_cluster_j

                        new_wcss = calculate_wcss(new_clusters)

                        if new_wcss < current_wcss:
                            adjusted_clusters = new_clusters
                            current_wcss = new_wcss
                            swapped = True

        # If no swaps were made, exit the loop
        if not swapped:
            break
    
    return adjusted_clusters




# no reclustering, leads to uneven groups
def solve1():
    distance_matrix = calc_distance_matrix(todays_words)
    word_groups = calc_clusters_kmeans(todays_words, distance_matrix)

    print("Word groups:")
    for group in word_groups:
        print(f"Group {group}: {word_groups[group]}")


# reclustering for groups of 4, minimising semantic_dist
def solve2():
    distance_matrix = calc_distance_matrix(todays_words)
    word_groups = calc_clusters_kmeans(todays_words, distance_matrix)

    # adjust clusters to ensure 4 groups of 4 words each
    adjusted_word_groups = adjust_clusters(word_groups, distance_matrix)

    print("Word groups:")
    for group in adjusted_word_groups:
        print(f"Group {group}: {adjusted_word_groups[group]}")


if __name__ == "__main__":
    # no reclustering
    solve1()

    # reclustering
    #solve2()
