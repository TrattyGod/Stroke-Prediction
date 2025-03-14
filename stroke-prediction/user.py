import os
import time
from termcolor import colored
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import cursor
import sys
import keyboard

class Model(nn.Module):
    def __init__(self, in_features=16, h1=32, h2=24, h3=12, h4=6, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid((self.out(x)))
        return x

def ClearScreen():
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')

def typewriter_effect(text, speed):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print() 

torch.manual_seed(41)

ClearScreen()

cursor.hide()

typewriter_effect(colored("Welcome to Stroke-Prediction V1.0.0", "green", attrs=["bold", "underline"]), 0.01)
input("Press [enter] to begin ")

user_data = []
question_num = 1
questions = ["Do you experience any chest pain?",
             "Do you experience shortness of breath?",
             "Do you ever have an irregular heartbeat?",
             "Do you suffer from fatigue or weakness?",
             "Do you experience dizziness?",
             "Do you have swelling (Edema)?",
             "Do you have pain in the neck/jaw/shoulders/back?",
             "Do you experience excessive sweating?",
             "Do you have a persistent cough?",
             "Do you experience nausea/vomiting?",
             "Do you have a high blood pressure?",
             "Do you have any chest discomfort (Activity)?",
             "Do you have cold hands/feet?",
             "Do you snore or have sleep apnea?",
             "Do you ever experience anxiety or have a feeling of doom?",
             "What is your age (21-100 for best results)?"]

explanations = [
    "Feeling an uncomfortable pressure, squeezing, or burning in your chest that can radiate to your back, neck, or arms.",
    "Struggling to get enough air, like you can't breathe deeply or fully.",
    "Noticing your heart skips beats, beats too fast, or flutters without any reason.",
    "Feeling unusually tired or weak, even after a good night's sleep or minimal activity.",
    "Experiencing lightheadedness, feeling faint, or losing your balance unexpectedly.",
    "Noticing puffiness or bloating, especially in your legs, ankles, and feet.",
    "Aches or discomfort in these areas that may or may not be connected to physical activity.",
    "Breaking out in a cold sweat without a clear cause, even when you're not hot or exerting yourself.",
    "Having a nagging cough that doesn't go away and isn't linked to a cold or flu.",
    "Feeling sick to your stomach or actually throwing up for no obvious reason.",
    "When your blood pressure readings are consistently higher than the normal range.",
    "Feeling tightness or pain in your chest specifically when you're active or exerting yourself.",
    "Having hands and feet that feel unusually cold compared to the rest of your body.",
    "Making loud, rattling noises while sleeping or experiencing interrupted breathing during sleep.",
    "Having a sudden, overwhelming sense of dread or anxiety that something bad is about to happen."
]

ClearScreen()

typewriter_effect(colored("Do not use Stroke Prediction to diagnose yourself. ALWAYS see a real doctor. This is just supposed to give a basic idea on your risks of stroke.", "yellow", attrs=["bold", "underline"]), 0.01)
input("Press [enter] to continue ")

ClearScreen()

typewriter_effect("Before we start, please enter your full name (your name will not be shared with anyone): ", 0.01)
name = input("> ")

ClearScreen()

for loop in range(15):
    print(f"Question [{question_num}/16] - {questions[question_num - 1]}")
    user = "N/A"

    while user != "y" and user != "n" and user != "yes" and user != "no"and user != "yeah"and user != "nah"and user != "1"and user != "0":
        print("Answer With Yes or No (? if you need help) ")
        user = input("> ").lower()

        if (user == "y" or user == "yes" or user == "yeah" or user == "1"):
            user_data.append(1.0)
        elif (user == "n" or user == "no" or user == "nah" or user == "0"):
            user_data.append(0.0)
        elif (user == "?" or user == "help"):
            print(colored(f"🛈 {explanations[question_num - 1]}", "blue", attrs=["bold", "underline"]))
            print()
        else:
            print(colored("⚠ Please Enter a Valid Answer", "red", attrs=["bold", "underline"]))
            print()

    print(f"Data Added [{question_num}/16] {"*" * question_num}{"-" * (16-question_num)}")
    time.sleep(0.5)
    
    ClearScreen()

    question_num += 1

print(f"Question [16/16] - {questions[15]}")

user = "N/A"

while not user.isdigit():
    print("Answer with a number. Do not include any letters ")
    user = input("> ")

    if (user.isdigit()):
        age = int(user)

        if (age < 1):
            age = 1.0
        if (age > 100):
            age = 100.0
        
        user_data.append(float(age))
    else:
        print(colored("⚠ Please Enter a Valid Answer", "red", attrs=["bold"]))

print("Data Added [16/16] ****************")
time.sleep(0.5)

ClearScreen()

for i in tqdm(range(100), desc="Loading Model", ascii=False, ncols=75):
    time.sleep(0.01)  

for i in tqdm(range(16), desc="Processing Case", ascii=False, ncols=75):
    time.sleep(0.2)

ClearScreen()

#user_data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]

loaded_model = Model()
loaded_model.load_state_dict(torch.load("MODEL.pt"))

user_case = torch.tensor(user_data)
with torch.no_grad():
    result = loaded_model(user_case.unsqueeze(0))
    result_binary = 1 if result > 0.5 else 0

print(colored("RESULTS", "green", attrs=["bold", "underline"]))
print()

current_date = datetime.date.today()
print(f"⧖ Date: {current_date}")
print(f"Name: {name}, Age: {user_data[15]}")

if (result_binary == 1):
    print("At Risk of Stroke: Yes")
else:
    print("At Risk of Stroke: No")

print(f"Risk Score: {int(result * 50)}/50")

confidence = result.item() * 100

if (confidence > 50):
    pass
if (confidence < 50):
    confidence = 100 - confidence

print(f"Confidence: {round(confidence, 2)}%", end="")

if (confidence <= 20):
    print(" (Weak)")
elif (confidence <= 40):
    print(" (OK)")
elif (confidence <= 60):
    print(" (Decent)")
elif (confidence <= 80):
    print(" (Strong)")
elif (confidence <= 100):
    print(" (Very Strong)")

print()
print()

print("Press [enter] to end the program. Feel free to use the program as many times as you wish ")

while True:
    if keyboard.is_pressed("enter"):
        break