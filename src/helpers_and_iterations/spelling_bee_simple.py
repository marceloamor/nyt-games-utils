""" -SPELLING BEE SOLVER-
- get dictionary list of english words
- ask for user input of outer letters and center letter
- return possible list of words, sorted by length and alphabetically
"""

with open("english_dict.txt") as f:
    english_dict = set(f.read().split("\n"))


def input_letters():
    while True:
        letters = input("Input the available letters, capitalise the center letter: \n")
        # strip spaces and non alpha characters

        center_letter = ""

        for letter in letters:
            if letter.isupper():
                center_letter = letter.lower()

        letters = set(letters.replace(" ", "").lower())

        if len(letters) == 7 and len(center_letter) == 1:
            break

    return (letters, center_letter)


def filter_dictionary(letters, center_letter):
    poss_words = []

    for word in english_dict:
        if center_letter not in word:
            continue
        if len(word) < 4:
            continue
        if not set(word).issubset(letters):
            continue
        poss_words.append(word)

    return poss_words  # .sort(key=len)


def solve():
    letters, center_letter = input_letters()

    poss_words = filter_dictionary(letters, center_letter)
    print("Possible Words:")
    # double sort for alpha and len sort cause why not
    for word in sorted(sorted(poss_words), key=len):
        print(word)
    print(f"Number of possible words: {len(poss_words)}")


if __name__ == "__main__":
    solve()
