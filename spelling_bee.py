""" -SPELLING BEE SOLVER-
- get dictionary list of english words
- ask for user input of outer letters and center letter
- return possible list of words, sorted by length and alphabetically
"""


class SpellingBeeSolver:
    def __init__(self, dictionary_file):
        self.english_dict = self.load_dictionary(dictionary_file)
        self.letters = set()
        self.center_letter = ""

    def load_dictionary(self, dictionary_file):
        with open(dictionary_file) as f:
            return set(f.read().split("\n"))

    def input_letters(self):
        while True:
            letters = input(
                "Input the available letters, capitalise the center letter: \n"
            )
            self.letters = set(letters.replace(" ", "").lower())
            self.center_letter = ""

            for letter in letters:
                if letter.isupper():
                    self.center_letter = letter.lower()

            if len(self.letters) == 7 and len(self.center_letter) == 1:
                break

    def filter_dictionary(self):
        poss_words = []

        for word in self.english_dict:
            if self.center_letter not in word:
                continue
            if len(word) < 4:
                continue
            if not set(word).issubset(self.letters):
                continue
            poss_words.append(word)

        return poss_words

    def solve(self):
        self.input_letters()
        poss_words = self.filter_dictionary()
        print("Possible Words:")
        for word in sorted(sorted(poss_words), key=len):
            print(word)
        print(f"Number of possible words: {len(poss_words)}")


if __name__ == "__main__":
    solver = SpellingBeeSolver("english_dict.txt")
    solver.solve()
