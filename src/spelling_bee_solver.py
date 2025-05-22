""" -SPELLING BEE SOLVER-
- fetch letters from NYT Spelling Bee website
- generate possible word combinations
- verify words against an online dictionary API
- return possible list of words, sorted by length and alphabetically
"""
import requests
from bs4 import BeautifulSoup
import itertools
from collections import defaultdict
import time


class SpellingBeeSolver:
    def __init__(self):
        self.letters = set()
        self.center_letter = ""
        self.valid_words = set()
        self.datamuse_cache = {}

    def fetch_todays_letters(self):
        """Fetch today's letters from the NYT Spelling Bee website."""
        try:
            url = "https://www.nytimes.com/puzzles/spelling-bee"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the game data in the page
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and "window.gameData" in script.string:
                    script_text = script.string
                    # Find the center letter
                    center_index = script_text.find('"centerLetter":"') + len('"centerLetter":"')
                    center_letter = script_text[center_index:center_index+1]
                    
                    # Find the outer letters
                    outer_index = script_text.find('"outerLetters":["') + len('"outerLetters":["')
                    outer_text = script_text[outer_index:outer_index+30]  # Grab enough characters
                    outer_letters = [c for c in outer_text if c.isalpha()][:6]  # Take first 6 alphabetic chars
                    
                    self.center_letter = center_letter.lower()
                    self.letters = set(outer_letters + [center_letter.lower()])
                    
                    print(f"Today's center letter: {center_letter.upper()}")
                    print(f"Today's letters: {''.join(outer_letters)}{center_letter.upper()}")
                    return True
            
            print("Could not find game data on the page.")
            return False
        except Exception as e:
            print(f"Error fetching today's letters: {e}")
            return False

    def input_letters(self):
        # First try to fetch today's letters
        if self.fetch_todays_letters():
            use_fetched = input("Use these letters? (y/n): ").lower()
            if use_fetched == 'y':
                return
        
        # If fetching failed or user wants to input manually
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

    def check_word_validity(self, word):
        """Check if a word is valid using the Datamuse API."""
        if word in self.datamuse_cache:
            return self.datamuse_cache[word]
        
        # Only check words that are at least 4 letters and contain the center letter
        if len(word) < 4 or self.center_letter not in word:
            self.datamuse_cache[word] = False
            return False
            
        try:
            # Use Datamuse API to check if word exists
            url = f"https://api.datamuse.com/words?sp={word}&md=d"
            response = requests.get(url)
            results = response.json()
            
            # If we get results and the first result exactly matches our word
            valid = len(results) > 0 and results[0]['word'] == word
            
            # Cache the result
            self.datamuse_cache[word] = valid
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.1)
            
            return valid
        except Exception as e:
            print(f"Error checking word '{word}': {e}")
            return False

    def generate_word_combinations(self):
        """Generate all possible word combinations from the letters."""
        all_combinations = []
        
        # Generate combinations of different lengths (4 to 15 letters)
        for length in range(4, 16):
            # Generate all permutations of the letters of the current length
            for combo in itertools.permutations(self.letters, length):
                word = ''.join(combo)
                # Check if the word contains the center letter
                if self.center_letter in word:
                    all_combinations.append(word)
        
        return all_combinations

    def find_valid_words(self):
        """Find all valid words from the letter combinations."""
        # Generate all possible combinations
        print("Generating word combinations...")
        combinations = self.generate_word_combinations()
        
        # Check each combination against the dictionary API
        print(f"Checking {len(combinations)} possible combinations...")
        valid_words = []
        
        # Use a progress counter
        total = len(combinations)
        checked = 0
        
        for word in combinations:
            if self.check_word_validity(word):
                valid_words.append(word)
            
            # Update progress
            checked += 1
            if checked % 100 == 0:
                print(f"Progress: {checked}/{total} ({(checked/total)*100:.1f}%)")
        
        return valid_words

    def solve(self):
        self.input_letters()
        
        # To make this more efficient, let's use Datamuse to search for words with our letters
        print("Searching for valid words using Datamuse API...")
        
        # Search for words containing the center letter
        url = f"https://api.datamuse.com/words?sp=*{self.center_letter}*&max=1000"
        response = requests.get(url)
        candidates = response.json()
        
        # Filter candidates to only include words with our letters
        valid_words = []
        for word_data in candidates:
            word = word_data['word']
            # Check if the word meets the criteria
            if (len(word) >= 4 and 
                self.center_letter in word and 
                set(word).issubset(self.letters)):
                valid_words.append(word)
        
        # Sort results by length and then alphabetically
        sorted_words = sorted(sorted(valid_words), key=len)
        
        # Group words by length
        word_groups = defaultdict(list)
        for word in sorted_words:
            word_groups[len(word)].append(word)
        
        # Display results
        print("\nPossible Words:")
        for length in sorted(word_groups.keys()):
            print(f"\n{length}-letter words:")
            for word in word_groups[length]:
                print(word)
        
        print(f"\nNumber of possible words: {len(sorted_words)}")


if __name__ == "__main__":
    solver = SpellingBeeSolver()
    solver.solve()
