from enum import Enum, unique
import pyautogui as pa

@unique
class Key_Type(Enum):
    LEFT = "left"
    RIGHT = "right"
    ALT = "alt"
    CTRL = "ctrl"
    TAB = "tab"
    WIN = "win"
    D = 'd'
    A = 'a'

keys_down = {key: False for key in Key_Type}


def key_up(k):
    keys_down[k] = False
    pa.keyUp(k.value)

def key_down(k: Key_Type):
    keys_down[k] = True
    pa.keyDown(k.value)

def press(k: Key_Type):
    pa.press(k.value)

def hot_key(k1,k2):
    pa.hotkey(k1.value,k2.value)