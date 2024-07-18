def wordle_solver(words: list[str]) -> list[str]:
    """
    Given a list of words, return a list of words that can be used to make a wordle.

    """
    # first two checks, that my regex doesnt guarantee
    for word in words:
        if len(word) != 5:
            raise ValueError("Word must be 5 characters long")
        if not word.isalpha():
            raise ValueError("Word must be all letters")

    # deal w double letters, and letters not in one location but definitely in another
    base_regex = "\b(?!)[a-zA-Z](?!)[a-zA-Z](?!)[a-zA-Z](?!)[a-zA-Z](?!)[a-zA-Z]\b"
    # ingest words

    # generate filters
    # filters = generate_filters(words)

    # iterate through word list excluding words w filters

    # return list of possible words

    # read in words.txt
    with open("words.txt", "r") as f:
        poss_words = f.read().splitlines()

    for word in poss_words:
        print(word)

    return words


# wordle_solver(["hello", "world"])


# generate filters
# def generate_filters(words: list[str]):
#     answer = ["." * 5]
#     for word in words:
#         for index, letter in enumerate(word):
#             # check if capital letter
#             if letter.isupper():
#                 pass


# filters needed:
# list of correctly placed green letter
# list of letters that are not in the word
# list of letters i.e. yellow that must be somewhere
#

# dict of {index: [letters]}
# list of letters that must be present
word_dict = {
    0: [letter for letter in "abcdefghijklmnopqrstuvwxyz"],
    1: [letter for letter in "abcdefghijklmnopqrstuvwxyz"],
    2: [letter for letter in "abcdefghijklmnopqrstuvwxyz"],
    3: [letter for letter in "abcdefghijklmnopqrstuvwxyz"],
    4: [letter for letter in "abcdefghijklmnopqrstuvwxyz"],
}

present_letters = []


def update_word_dict(
    word_dict: dict, present_letters: list[str], guessed_words: list[str]
):
    # iterate through guessed words and update known letters
    for word in guessed_words:
        for index, letter in enumerate(word):
            if letter.isupper():
                # update dict at index
                word_dict[index] = [letter]
                present_letters.append(letter)
            if letter.islower():
                # remove letter from dict at all indices
                for index in word_dict:
                    word_dict[index].remove(letter)
            if letter == "!":
                # that means next letter is yellow
                present_letters.append(word[index + 1])
    print(word_dict)
    print(present_letters)

    # for index, letter in enumerate(word):
    #     if letter.isupper():
    #         word_dict[index] = [letter]
    #     else:
    #         word_dict[index].remove(letter)
    #         present_letters.append(letter)


update_word_dict(word_dict, present_letters, ["hELps"])
