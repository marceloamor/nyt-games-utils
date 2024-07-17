""" -LETTERBOXED SOLVER-
- get dictionary list of english words
- ask for user input of each side of letters
- return a possible list of words
"""

with open("english_dict.txt") as f:
    english_dict = set(f.read().split("\n"))


def input_letters():
    while True:
        letters = input(
            "Input the available letters, with a hyphen between sides, ie. ABC-DEF-GHI-JKL: \n"
        )
        # format input
        letters = letters.replace(" ", "").lower()

        groupA, groupB, groupC, groupD = letters.split("-")
        # check each group is of length 3
        if len(groupA) != 3 or len(groupB) != 3 or len(groupC) != 3 or len(groupD) != 3:
            print("Error - invalid input, please try again")
            continue

        letters = set(letters)
        letters.remove("-")
        print(letters)

        if len(letters) == 12:
            break

    return letters


def filter_dictionary(letters):
    poss_words = []

    for word in english_dict:
        if not word.isalpha():
            continue
        print(word)
        if len(word) < 3:
            continue
        print(word)
        if not set(word).issubset(letters):
            continue
        print(word)
        poss_words.append(word)

    print(poss_words.sort(key=len))
    return poss_words.sort(key=len)


def solve():
    letters, center_letter = input_letters()

    poss_words = filter_dictionary(letters, center_letter)
    print("Possible Words:")
    # double sort for alpha and len sort cause why not
    for word in sorted(sorted(poss_words), key=len):
        print(word)
    print(f"Number of possible words: {len(poss_words)}")


if __name__ == "__main__":
    # solve()
    letters = input_letters()
    filter_dictionary(letters)
