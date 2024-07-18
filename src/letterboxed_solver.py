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



class LetterboxedSolver:
    def __init__(self, dictionary_file):
        self.dictionary = self.load_dictionary(dictionary_file)
        self.possible_words = self.dictionary.copy()

    def load_dictionary(self, dictionary_file):
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
    solver = LetterboxedSolver("english_dict.txt")
    solver.solve()
