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
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time



class StrandsSolver:
    def __init__(self):
        self.letter_matrix = self.load_todays_matrix()

    def load_todays_matrix(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--no-sandbox")  # Add this option to avoid some potential issues
        options.add_argument("--disable-dev-shm-usage")  # Add this option to avoid some potential issues

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.get("https://www.nytimes.com/games/strands")
        time.sleep(5)

        try:
            board_div = driver.find_element(By.CLASS_NAME, "UOpmtW_board")
            print(board_div)
            letter_divs = board_div.find_elements(By.CLASS_NAME, "pRjvKq_item")

            letters = [div.text for div in letter_divs]

            print(letters)
        except Exception as e:
            print(e)


    def load_todays_matrix_old(self):
            # pull dictionary from git repo to get around 1 file limit on mobile IDE
        response = requests.get("https://www.nytimes.com/games/strands")
        # get the divs in class 'UOpmtW_board' class
        # pull the letters from each div, with the class 'pRjvKq_item'
        # create a 6x8 matrix with the letters
        # return the matrix
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            print(soup)
            board = soup.find("div", class_="UOpmtW_board")
            print(board)
        return board


    def interface(self, guess_num):
        def print_boxed_message(message):
            print("╔" + "═" * (len(message) + 2) + "╗")
            print("║ " + message + " ║")
            print("╚" + "═" * (len(message) + 2) + "╝")

        def print_boxed_input(prompt):
            print("╭" + "─" * (len(prompt) + 2) + "╮")
            response = input("│ " + prompt + " │\n╰" + "─" * (len(prompt) + 2) + "╯\n> ")
            return response

        if guess_num == 1:
            guess = print_boxed_input("As usual, I propose starting with 'crane', but what is your first guess?")
            feedback = print_boxed_input(f"Enter feedback for {guess}. (G for green, Y for yellow, X for black/grey)")
            self.process_feedback(guess, feedback)
        else:
            guess = self.make_guess()
            print_boxed_message(f"My guess #{guess_num} would be: {guess}")
            feedback = print_boxed_input("Enter feedback (G for green, Y for yellow, X for black/grey)")
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
    solver = StrandsSolver()
    # solver = WordleSolver("mobile")
    print(solver.letter_matrix)
