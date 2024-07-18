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
        return words
    







    ##################

    def process_feedback(self, guess, feedback):
        self.possible_words = self.filter_words(guess, feedback)

    def filter_words(self, guess, feedback):
        filtered_words = []
        for word in self.possible_words:
            if self.is_valid_word(word, guess, feedback):
                filtered_words.append(word)
        return filtered_words

    def is_valid_word(self, word, guess, feedback):
        for i in range(5):
            if feedback[i] == "G" and word[i] != guess[i]:
                return False
            if feedback[i] == "Y" and (word[i] == guess[i] or guess[i] not in word):
                return False
            if feedback[i] == "X" and guess[i] in word:
                return False
        return True

    def interface(self, guess_num):
        if guess_num == 1:
            guess = input(
                "As usual, I propose starting with 'crane', but what is your first guess?"
            )
            feedback = input(
                f"Enter feedback for {guess}. (G for green, Y for yellow, X for black/grey): "
            )
            self.process_feedback(guess, feedback)
        else:
            guess = self.make_guess()
            print(f"My guess #{guess_num} would be: {guess}")
            feedback = input(
                "Enter feedback (G for green, Y for yellow, X for black/grey): "
            )
            self.process_feedback(guess, feedback)
        return guess

    def solve(self):
        for i in range(1, 6):
            next_guess = self.interface(i)
            if next_guess == None:
                print("No more possible words")
                break
            if next_guess == "hello":
                print("I guessed the word!")
                break


if __name__ == "__main__":
    solver = ConnectionsSolver("src/dictionaries/english_dict.txt")
    solver.request_todays_words()
