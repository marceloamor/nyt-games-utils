""" -NYT CONNECTIONS SOLVER-

alright lets do some planning here before we start coding


"""
import spacy
import numpy as np
from sklearn.cluster import KMeans
import requests
from icecream import ic
from datetime import datetime

nlp = spacy.load("en_core_web_md")


class ConnectionsSolver:
    def __init__(self, dictionary_file):
        self.dictionary = self.load_dictionary(dictionary_file)
        self.todays_words = self.request_todays_words()
        self.semantic_distance_matrix = self.create_semantic_distance_matrix()
        self.word_groups = self.calc_clusters_kmeans()

    def load_dictionary(self, dictionary_file):
        with open(dictionary_file) as f:
            return [word.strip() for word in f if len(word.strip()) == 5]

    def request_todays_words(self):
        today = datetime.today().strftime("%Y-%m-%d")
        response = requests.get(
            f"https://www.nytimes.com/svc/connections/v1/{today}.json"
        )
        response.raise_for_status()
        word_collection = response.json()["startingGroups"]
        words = [word.lower() for array in word_collection for word in array]
        ic(words)
        return words

    def calculate_semantic_difference(self, word1, word2):
        tokens = nlp(f"{word1} {word2}")

        token1, token2 = tokens[0], tokens[1]
        return token1.similarity(token2)

    def create_semantic_distance_matrix(self):
        n = len(self.todays_words)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.calculate_semantic_difference(
                    self.todays_words[i], self.todays_words[j]
                )
        ic(matrix)
        return matrix

    def calc_most_likely_cluster(self):
        # create a group of 4 words that minimise intra-group semantic distance

        def calc_group_distance(group):
            return sum(
                [
                    self.calculate_semantic_difference(word1, word2)
                    for word1 in group
                    for word2 in group
                ]
            )

        def calc_most_likely_clusters_helper(words, groups):
            if not words:
                return groups
            best_group = None
            best_distance = float("inf")
            for group in groups:
                distance = calc_group_distance(group + [words[0]])
                if distance < best_distance:
                    best_distance = distance
                    best_group = group
            best_group.append(words[0])
            return calc_most_likely_clusters_helper(words[1:], groups)

        cluster = calc_most_likely_clusters_helper(
            self.todays_words, [[] for _ in range(4)]
        )
        ic(cluster)
        return cluster

    def calc_clusters_kmeans(self):
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(self.semantic_distance_matrix)
        cluster_labels = kmeans.labels_
        word_groups = {}
        for i in range(num_clusters):
            word_groups[i] = [
                self.todays_words[j]
                for j, label in enumerate(cluster_labels)
                if label == i
            ]
        ic(word_groups)
        return word_groups

    def adjust_clusters(self):
        kmeans = KMeans(n_clusters=len(self.word_groups))
        kmeans.fit(self.semantic_distance_matrix)
        initial_wcss = kmeans.inertia_

        def calculate_wcss(clusters):
            wcss = 0
            for cluster in clusters:
                cluster_center = np.mean(
                    [
                        self.calculate_semantic_difference(word1, word2)
                        for word1 in cluster
                        for word2 in cluster
                    ]
                )
                wcss += sum(
                    [
                        (self.calculate_semantic_difference(word, cluster_center)) ** 2
                        for word in cluster
                    ]
                )
            return wcss

        def adjust_clusters_helper(clusters, wcss):
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    if i != j:
                        temp_clusters = clusters.copy()
                        temp_clusters[j].append(temp_clusters[i].pop())
                        temp_wcss = calculate_wcss(temp_clusters)
                        if temp_wcss < wcss:
                            return adjust_clusters_helper(temp_clusters, temp_wcss)
            return clusters

        new_clusters = adjust_clusters_helper(
            list(self.word_groups.values()), initial_wcss
        )
        ic(new_clusters)
        return new_clusters

    ##################


if __name__ == "__main__":
    solver = ConnectionsSolver("src/dictionaries/english_dict.txt")
    solver.calc_most_likely_cluster()
