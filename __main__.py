from LoanBot import LoanBot
import csv

loanBot = LoanBot()

def display_menu():
    choice = 0
    while choice != 4:
        try:
            print("Welcome to the AI Loan Application Service! What would you like to do?")
            print("1. Check to see if you qualify!")
            print("2. Train the model with a file")
            print("3. Test the model with a file")
            print("4. Quit")
            choice = int(input())

            if choice == 1:
                try_qualify()
            elif choice == 2:
                train_model()
            elif choice == 3:
                test_model()
            elif choice != 4:
                print("Please enter a valid choice.")

        except ValueError:
            print("Choice must be a number.")

def try_qualify():
    print("Excellent! I will ask you some questions, and then I will tell you whether or not you qualify!")

    try:
        age = int(input("How old are you?\n"))
        yearly_income = int(input("What is your yearly income (after taxes)?\n"))
        credit_score = int(input("What is your current credit score?\n"))
        desired_amt = int(input("What is the desired amount for the loan?\n"))

        print("Alright, let's see if you qualify...")
        result = loanBot.predict([age, yearly_income, credit_score, desired_amt])
        
        if result == 1:
            print("\n--- Congratulations! You qualify! ---\n")
        else:
            print("\n I'm sorry! You don't qualify for a loan of $" + str(desired_amt) + " at this time. \n")

    except ValueError:
        print("Your answers must be numbers.")

def train_model():
    file_name = input("Please specify the name of file with the training data: ")

    try:
        with open(file_name, mode = 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skipping header
            data = []
            for line in reader:
                line = [int(item) for item in line]
                data.append(line)

        print("Training...")
        loanBot.train(data)
        print("Done!")

    except FileNotFoundError:
        print("The file does not exist.")
    except ValueError:
        print("Some of the data in the file couldn't be converted to integers.")

def test_model():
    file_name = input("Please specify the name of file with the test data: ")

    try:
        with open(file_name, mode = 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skipping header
            data = []
            for line in reader:
                line = [int(item) for item in line]
                data.append(line)

        print("Testing...")
        loanBot.test(data)
        print("Done!")

    except FileNotFoundError:
        print("The file does not exist.")
    except ValueError:
        print("Some of the data in the file couldn't be converted to integers.")

display_menu()