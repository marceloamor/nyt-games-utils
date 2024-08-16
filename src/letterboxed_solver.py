""" -LETTERBOXED SOLVER-

Rules:
1. The word must be at least 3 letters long
2. Consecutive letters must not be from the same side
3. The last letter of a word becomes the first in the next word
4. 

alright lets do some planning here before we start coding

-create a Side class containing the letters of the side



solving ideas:
- immediately eliminate words that contain the same letter twice or is not a subset of all the given letters
- maybe also eliminate words that contain two letters from the same side consecutively  
    - do these two with a __word_is_valid__(word) method
        - checks word length
        - checks word is a subset of the given letters
        - checks for consecutive letters from the same side

"""

import requests
from bs4 import BeautifulSoup
import json
import time

from icecream import ic


class LetterboxedSolver:
    def __init__(self, solve_mode: str):
        (
            self.letters,
            self.poss_words,
            self.par,
            self.best_solution,
        ) = self.fetch_todays_game_data()
        self.solve_mode = solve_mode

    def load_dictionary(self, dictionary_file):
        with open(dictionary_file) as f:
            return [word.strip() for word in f if len(word.strip()) == 5]

    def fetch_todays_game_data(self):
        url = "https://www.nytimes.com/puzzles/letter-boxed"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to load page {url}")

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the script tag containing the game data
        script_tag = soup.find("script", text=lambda t: t and "window.gameData" in t)
        if not script_tag:
            raise Exception("Game data script tag not found")

        # Extract the JSON data from the script content
        script_content = script_tag.string
        json_data_start = script_content.find("window.gameData = ") + len(
            "window.gameData = "
        )
        json_data_end = script_content.find(";", json_data_start)
        json_data = script_content[json_data_start:json_data_end]

        try:
            game_data = json.loads(json_data + "}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")

        todays_letters = game_data["sides"]
        poss_words = game_data["dictionary"]
        par = game_data["par"]
        best_solution = game_data["ourSolution"]

        return (todays_letters, poss_words, par, best_solution)

    def cheat_interface(self):
        print("Fetching today's game data...")
        time.sleep(1)
        print("Todays letters are: ", self.letters)
        time.sleep(0.5)
        print("Calculating optimal solution...")
        time.sleep(1)
        print("...")
        time.sleep(1)
        print("...")
        time.sleep(1)
        print("...")
        time.sleep(1)
        print(
            f"The best solution is: {self.best_solution}, with a score of {len(self.best_solution)}, on a par-{self.par} puzzle"
        )

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
    solver = LetterboxedSolver("cheat_mode")
    if solver.solve_mode == "cheat_mode":
        solver.cheat_interface()
    else:
        print("Full throated attempt under construction...")
