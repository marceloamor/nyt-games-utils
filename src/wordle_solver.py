""" -WORDLE SOLVER-

alright lets do some planning here before we start coding

1. load the dictionary
2. ask for the first guessed word
3. ask for the arrangement of correct letters
4. use simple letter frequency to determine the n best next guesses (will optimise this algo later)
5. repeat steps 2-4 until the word is guessed
6. profit

"""
import requests

letter_counts = {
    "a": 906,
    "b": 266,
    "c": 446,
    "d": 370,
    "e": 1053,
    "f": 206,
    "g": 299,
    "h": 377,
    "i": 646,
    "j": 27,
    "k": 202,
    "l": 645,
    "m": 298,
    "n": 548,
    "o": 672,
    "p": 345,
    "q": 29,
    "r": 835,
    "s": 617,
    "t": 667,
    "u": 456,
    "v": 148,
    "w": 193,
    "x": 37,
    "y": 416,
    "z": 35,
}


class WordleSolver:
    def __init__(self, dictionary_file):
        self.dictionary = self.load_dictionary(dictionary_file)
        self.possible_words = self.dictionary.copy()

    def load_dictionary(self, dictionary_file):
        if dictionary_file == "mobile":
            # pull dictionary from git repo to get around 1 file limit on mobile IDE
            req = requests.get(
                "https://raw.githubusercontent.com/marceloamor/nyt-games-utils/main/src/dictionaries/wordle_words.txt"
            )
            if req.status_code == 200:
                content = req.text
                words = content.splitlines()
                return [word.strip() for word in words if len(word.strip()) == 5]
        else:
            with open(dictionary_file) as f:
                return [word.strip() for word in f if len(word.strip()) == 5]

    def make_guess(self):
        # rank words based on letter frequency
        # count repeated letters once only
        best_guess = ""
        best_score = 0
        if self.possible_words:
            for word in self.possible_words:
                score = sum(letter_counts[letter] for letter in set(word))
                if score > best_score:
                    best_score = score
                    best_guess = word

            return best_guess
        return None

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
        def print_boxed_message(message):
            print("╔" + "═" * (len(message) + 2) + "╗")
            print("║ " + message + " ║")
            print("╚" + "═" * (len(message) + 2) + "╝")

        def print_boxed_input(prompt):
            print("╭" + "─" * (len(prompt) + 2) + "╮")
            response = input(
                "│ " + prompt + " │\n╰" + "─" * (len(prompt) + 2) + "╯\n> "
            )
            return response

        if guess_num == 1:
            guess = print_boxed_input(
                "As usual, I propose starting with 'crane', but what is your first guess?"
            )
            feedback = print_boxed_input(
                f"Enter feedback for {guess}. (G for green, Y for yellow, X for black/grey)"
            )
            self.process_feedback(guess, feedback)
        else:
            guess = self.make_guess()
            print_boxed_message(f"My guess #{guess_num} would be: {guess}")
            feedback = print_boxed_input(
                "Enter feedback (G for green, Y for yellow, X for black/grey)"
            )
            if feedback.lower() == "ggggg":
                return "guessed it!"
            self.process_feedback(guess, feedback)

        return guess

    def solve(self):
        for i in range(1, 7):
            next_guess = self.interface(i)
            if next_guess == None:
                print("No more possible words")
                break
            if next_guess == "hello":
                print("I guessed the word!")
                break


if __name__ == "__main__":
    solver = WordleSolver("src/dictionaries/wordle_words.txt")
    # solver = WordleSolver("mobile")
    solver.solve()
