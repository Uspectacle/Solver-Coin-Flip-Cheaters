import time
import threading
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode
import numpy as np
import pytesseract
import cv2
from PIL import ImageGrab
import pyautogui

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

prob_fair = 0.5
prob_cheat = 0.75
win_reward = 15
loss_reward = -30 
prior_prior = 0.5
prob_coin = 0.5
delay = 0.1
best_aversion = 1.1
best_biais = 3.2

data = {}
past_toss = 100
past_state = []
past_choice = 2

def binomial(state, prob_head):
    return prob_head**state[0] * (1 - prob_head)**state[1]

def posterior_cheat(state, prior_cheat):
    return (prior_cheat * binomial(state, prob_cheat) 
    / (prior_cheat * binomial(state, prob_cheat) 
    + (1 - prior_cheat) * binomial(state, prob_fair)))

def expected(prob_cheat, aversion):
    reward_cheat = prob_cheat * win_reward + (1 - prob_cheat) * loss_reward * aversion
    reward_fair = (1 - prob_cheat) * win_reward + prob_cheat * loss_reward * aversion
    return max(reward_cheat, reward_fair)

def prob_head(prob_cheat):
    return prob_cheat * prob_cheat + prob_fair * (1 - prob_cheat)

def expected_next(state, prior_cheat, aversion):
    expected_head = expected(posterior_cheat([1, 0], prior_cheat), aversion)
    expected_tails = expected(posterior_cheat([0, 1], prior_cheat), aversion)
    return expected_head * prob_head(prob_cheat) + expected_tails * (1 - prob_head(prob_cheat))

def rand(prob):
    return int(np.random.rand() > prob)

def choose(state, aversion, biais, no_toss = False):
    prob = posterior_cheat(state, prior_prior)
    reward = expected(prob, aversion)
    next_reward = expected_next(state, prob, aversion) - 1
    if (reward > next_reward and reward > biais) or no_toss:
        return (int(prob * win_reward + (1 - prob) * loss_reward 
        < (1 - prob) * win_reward + prob * loss_reward))
    else:
        return 2

def bbox_to_text(x_1, y_1, x_2, y_2):
    cap = ImageGrab.grab(bbox =(x_1, y_1, x_2, y_2))
    return pytesseract.image_to_string(
        cv2.cvtColor(np.array(cap), cv2.COLOR_BGR2GRAY), lang ='eng')

def get_state():
    texts = bbox_to_text(1420, 425, 1530, 480).split()
    if len(texts) == 4:
        if texts[0] == "Heads:" and texts[2] == "Tails:":
            return [int(texts[1]), int(texts[3])]
    return None

def get_flip():
    texts = bbox_to_text(1450, 650, 1620, 680).split()
    if len(texts) == 3:
        if texts[1] == "Flips" and texts[2] == "left":
            try:
                return int(texts[0])
            except:
                return None
    return None

def is_end():
    texts = bbox_to_text(1200, 500, 1570, 570).split()
    return texts == ['Save', 'your', 'score', 'to', 'the', 'leaderboard?']

def flip():
    pyautogui.click(1350, 700)

def label_fair():
    pyautogui.click(1350, 790)

def label_cheat():
    pyautogui.click(1530, 790)

def submit():
    pyautogui.click(1440, 715)

def replay():
    pyautogui.click(1440, 780)


def game(aversion, biais):
    toss = 100
    data = [[0, 0], [0, 0]]
    success = 0

    while(toss>0):
        coin = rand(prob_coin)
        true_prob = prob_fair if coin else prob_cheat
        state = [0, 0]
        while(True):
            choice = choose(state, aversion, biais, toss <= 0)
            if choice == 2:
                toss += -1
                state[rand(true_prob)] += 1
            else:
                data[coin][choice] += 1
                if coin == choice:
                    success += 1
                    out = 'WINS'
                    toss += win_reward
                else: 
                    out = 'LOST'
                    toss += loss_reward
                break
    return success


def action(choice):
    if choice == 2:
        flip()
        print("Action: Flip")
        return
    elif choice == 1:
        label_fair()
        print("Action: Label Fair")
        return
    elif choice == 0:
        label_cheat()
        print("Action: Label Cheat")
        return

def data_update(state, choice, reward):
    if str(state) == '[]':
        return
    if choice == 0:
        choice = 'Cheat'
    elif choice == 1:
        choice = 'Fair '
    else:
        return
    if reward > 7:
        reward = 'True '
    elif reward < -7:
        reward = 'False'
    else:
        return
    if str(state) not in data.keys():
        data[str(state)] = {choice: {'True ': 0, 'False': 0}}
    if choice not in data[str(state)].keys():
        data[str(state)][choice] = {'True ': 0, 'False': 0}
    data[str(state)][choice][reward] += 1
    print(reward, choice)
    return

def play(past_state, past_choice, past_toss):
    if is_end():
        submit()
        time.sleep(3)
        replay()
        return (past_state, past_choice, past_toss)

    toss = get_flip()
    if toss == None:
        return (past_state, past_choice, past_toss)
    if toss != past_toss:
        data_update(past_state, past_choice, toss - past_toss)
        past_toss = toss
    if toss < 0:
        return (past_state, past_choice, past_toss)

    state = get_state()
    if state == None:
        action(2)
        return (past_state, past_choice, past_toss)

    print(f'state : {state}')
    if state == past_state:
        return (past_state, past_choice, past_toss)
    past_state = state
    choice = choose(state, best_aversion, best_biais, toss == 0)
    if state == [4, 4]:
        choice = 2
    past_choice = choice
    action(choice)
    return (past_state, past_choice, past_toss)

def print_data():
    print('----------------')
    for key, item in data.items():
        print(f'{key}: \t{item}')
    print('----------------')


start_stop_key = KeyCode(char='s')
exit_key = KeyCode(char='e')
print_key = KeyCode(char='p')

class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True
        self.past_state, self.past_choice, self.past_toss = past_state, past_choice, past_toss

    def start_clicking(self):
        print('--- START ---')
        self.running = True

    def stop_clicking(self):
        print('--- STOP ---')
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            index = 0
            while self.running:
                index += 1
                if not index % 100:
                    print_data()
                self.past_state, self.past_choice, self.past_toss = play(self.past_state, self.past_choice, self.past_toss)
                time.sleep(delay)
            time.sleep(delay)

click_thread = ClickMouse()
click_thread.start()

def on_press(key):
    if key == start_stop_key:
        if click_thread.running:
            click_thread.stop_clicking()
        else:
            click_thread.start_clicking()
    elif key == exit_key:
        click_thread.exit()
        listener.stop()
    elif key == print_key:
        print_data()


with Listener(on_press=on_press) as listener:
    listener.join()


