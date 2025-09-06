import random

responses = {
    "hello": ["Hi there!", "Hello 👋", "Hey! How can I help you?"],
    "bye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Anytime!"]
}

while True:
    user = input("You: ").lower()
    if user in responses:
        print("AI Bot:", random.choice(responses[user]))
    elif user == "exit":
        print("AI Bot: Bye! 👋")
        break
    else:
        print("AI Bot: Sorry, I don’t understand.")
