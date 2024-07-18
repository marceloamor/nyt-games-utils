import string
from pathlib import Path


with open("wordle_words.txt", "r") as f:
    WORDS = f.read().splitlines()
DICT = "/usr/share/dict/american-english"

ALLOWABLE_CHARACTERS = set(string.ascii_letters)
ALLOWED_ATTEMPTS = 6
WORD_LENGTH = 5

# WORDS = {
#     word.lower()
#     for word in Path(DICT).read_text().splitlines()
#     if len(word) == WORD_LENGTH and set(word) < ALLOWABLE_CHARACTERS
# }


def input_word():
    while True:
        word = input("Input the word you entered> ")
        if len(word) == WORD_LENGTH and word.lower() in WORDS:
            break
    return word.lower()


def input_response():
    print("Type the color-coded reply from Wordle:")
    print("  g for Green")
    print("  y for Yellow")
    print("  ? for Gray")
    while True:
        response = input("Response from Wordle> ").lower()
        if len(response) == WORD_LENGTH and set(response) <= {"g", "y", "?"}:
            break
        else:
            print(f"Error - invalid answer {response}")
    return response


# word_vector = [set(string.ascii_lowercase) for _ in range(WORD_LENGTH)]


def match_word_vector(word, word_vector):
    assert len(word) == len(word_vector)
    for letter, v_letter in zip(word, word_vector):
        if letter not in v_letter:
            return False
    return True


def match(word_vector, possible_words):
    return [word for word in possible_words if match_word_vector(word, word_vector)]


def display_word_table(possible_words):
    for word in possible_words:
        print(word)
    print()


def solve():
    possible_words = WORDS.copy()
    word_vector = [set(string.ascii_lowercase) for _ in range(WORD_LENGTH)]
    for attempt in range(1, ALLOWED_ATTEMPTS + 1):
        print(f"Attempt {attempt} with {len(possible_words)} possible words")
        if attempt != 1:
            display_word_table(possible_words)
        word = input_word()
        response = input_response()
        for idx, letter in enumerate(response):
            if letter == "G":
                word_vector[idx] = {word[idx]}
            elif letter == "Y":
                try:
                    word_vector[idx].remove(word[idx])
                except KeyError:
                    pass
            elif letter == "?":
                for vector in word_vector:
                    try:
                        vector.remove(word[idx])
                    except KeyError:
                        pass
        possible_words = match(word_vector, possible_words)


if __name__ == "__main__":
    solve()
