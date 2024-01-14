import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from typing import Tuple, Union
import math
import pyautogui as pa
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import threading
coordinates = [[] for i in range(21)]
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
RECOGNIZED = False
TRACE_TIME = 0.4
LAZY = 10
LEFT = "L"
DOWN = "D"
RIGHT = "R"
UP = "U"
MID = "M"
FRONT = "F"
BACK = "B"
simulation_x = None
simulation_y = None
simulation_z = None
definitions_move_x = []
definitions_move_y = []
definitions_move_z = []
lock = False
lock_time = 0
duration_lock = 1.5
class fuzzyThread (threading.Thread):
    def __init__(self, threadID, name, axis):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.axis = axis

    def run(self):
        definition_move = fuzzy(self.axis)
        if self.axis == 'x':
            global definitions_move_x
            definitions_move_x = definition_move
        elif self.axis == 'y':
            global definitions_move_y
            definitions_move_y = definition_move
        elif self.axis == 'z':
            global definitions_move_z
            definitions_move_z = definition_move

def calculate_pluse(x1, x2)->float:
    return x2-x1

def clear():
    global coordinates
    coordinates = [[] for i in range(21)]


def result_generate(result:[]):
    if len(result) == 0:
        return MID
    count = {}
    for i in result:
        if i not in count.keys():
            count[i] = 1
        else:
            count[i] = count[i] + 1
    move = result[0]
    for i in count.keys():
        if count[i]>count[move]:
            move = i
    return move
'''
创建一个基于fuzzy逻辑的分类器
A,B,C是三个连续的状态，例如LMR
'''
def simulation_init(A,B,C):
    values = np.arange(-10,10,1)
    k_values = ctrl.Antecedent(values,"k")
    k_values[A] = fuzz.trimf(k_values.universe, [-10, -10, 0])
    k_values[B] = fuzz.trimf(k_values.universe, [-3, 0, 3])
    k_values[C] = fuzz.trimf(k_values.universe, [0, 10, 10])

    # 定义输出变量
    state = np.arange(0, 11, 1)
    movement_state = ctrl.Consequent(state, 'move')
    # 定义运动的方向的模糊集和隶属函数
    movement_state[A] = fuzz.trimf(movement_state.universe, [0, 0, 5])
    movement_state[B] = fuzz.trimf(movement_state.universe, [0, 5, 10])
    movement_state[C] = fuzz.trimf(movement_state.universe, [5, 10, 10])
    rule1 = ctrl.Rule(k_values[A],movement_state[A])
    rule2 = ctrl.Rule(k_values[B],movement_state[B])
    rule3 = ctrl.Rule(k_values[C],movement_state[C])
    # 创建控制系统
    system = ctrl.ControlSystem([rule1, rule2, rule3])

    # 创建模糊逻辑引擎
    simulation = ctrl.ControlSystemSimulation(system)
    # 绘制模糊逻辑输出
    ## movement_state.view(sim=simulation)
    return simulation

def simulations_init():
    global simulation_x
    global simulation_y
    global simulation_z
    simulation_x = simulation_init(LEFT,MID,RIGHT)
    simulation_y = simulation_init(UP,MID,DOWN)
    simulation_z = simulation_init(FRONT,MID,BACK)


def separate_coordinates(axis=''):
    global coordinates
    x_labels = []
    y_labels =[]
    z_labels = []
    for i in coordinates:
        figure_x = []
        figure_y = []
        figure_z = []
        for j in i:
            figure_x.append(j[0])
            figure_y.append(j[1])
            figure_z.append(j[2])
        x_labels.append(figure_x)
        y_labels.append(figure_y)
        z_labels.append(figure_z)
    if axis == 'x':
        return x_labels
    if axis == 'y':
        return y_labels
    if axis == 'z':
        return z_labels
    return x_labels, y_labels, z_labels

def map_output_to_state(output_value, axis='x'):
    if output_value <= 2.5:
        if axis == 'x':
            return LEFT
        if axis == 'y':
            return UP
        if axis == 'z':
            return FRONT
    elif output_value <= 7.5:
        return MID
    else:
        if axis == 'x':
            return RIGHT
        if axis == 'y':
            return DOWN
        if axis == 'z':
            return BACK

def fuzzy(axis = 'x')->[]:
    global coordinates
    labels = separate_coordinates(axis)
    simulation = None
    if axis == 'x':
        global simulation_x
        simulation = simulation_x
    elif axis == 'y':
        global simulation_y
        simulation = simulation_y
    elif axis == 'z':
        global simulation_z
        simulation = simulation_z
    result = []
    figure_gaps = []
    for i in range(len(labels)):
        gap = []
        for l in range(len(labels[i])-1):
            g = calculate_pluse(labels[i][l],labels[i][l+1])
            gap.append(g)
        figure_gaps.append(gap)
    for i in figure_gaps:
        r = []
        for k in i:
            simulation.input["k"] = k
            simulation.compute()
            r.append(map_output_to_state(simulation.output['move'],axis))
        result.append(r)
    return [result_generate(m) for m in result]


def read_move():
    thread_x = fuzzyThread(1, 'x thread', 'x')
    thread_y = fuzzyThread(2, 'y thread', 'y')
    thread_z = fuzzyThread(3, 'z thread', 'z')
    thread_x.start()
    thread_y.start()
    thread_z.start()
    thread_x.join()
    thread_y.join()
    thread_z.join()

def operation(results):
    global RECOGNIZED
    global coordinates
    global lock
    global lock_time
    if lock:
        return
    if results.multi_hand_landmarks:
        if not RECOGNIZED:
            RECOGNIZED = True
        read_move()
        if definitions_move_x[0] == MID and definitions_move_x[8] == RIGHT and definitions_move_x[12] == RIGHT and definitions_move_x[16] == RIGHT:
            pa.hotkey('alt', 'tab')
            lock = True
            lock_time = time.time()
        elif definitions_move_x[0] == MID and definitions_move_x[8] == LEFT and definitions_move_x[12] == LEFT and definitions_move_x[16] == LEFT:
            pa.hotkey('alt', 'tab','left','left')
            lock = True
            lock_time = time.time()
        elif definitions_move_y[0] == MID and definitions_move_y[8] == DOWN and definitions_move_y[12] == DOWN and definitions_move_y[16] == DOWN:
            pa.hotkey('win', 'd')
            lock = True
            lock_time = time.time()
        ## test([0, 12, 14, 16],definitions_move_y)
    else:
        if RECOGNIZED:
            clear()
            RECOGNIZED = False

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, normalized_z: float, image_width: int,
    image_height: int) -> Union[None, Tuple[float, float, float]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))
  z_px = min(math.floor(normalized_z * image_width), image_width - 1)
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px,z_px

def test(dots:[], results:[]):
    for i in dots:
        print(results[i], end=" ")
    print("")

def draw_image(image,results):
    global coordinates
    for hand_landmark in results.multi_hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark.landmark
        ])
        image_rows, image_cols, _ = image.shape
        print(image_rows,image_cols)
        ##covert to coordinate in image.
        solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return image

def read_coordinates(image,results)->None:
    global coordinates
    for hand_landmark in results.multi_hand_landmarks:
        image_rows, image_cols, _ = image.shape
        ##covert to coordinate in image.
        for idx in range(len(hand_landmark.landmark)):
            landmark = hand_landmark.landmark[idx]
            x,y,z = normalized_to_pixel_coordinates(landmark.x,landmark.y,landmark.z, image_cols, image_rows)
            # print("%.4f, %.4f, %.4f"%(x,y,z));
            timestamp = time.time()  # 记录生成坐标的时间戳
            # 保留过去一秒内的坐标
            coordinates[idx].append([x, y, z, timestamp])
            coordinates[idx] = [(x, y, z, t) for x, y, z, t in coordinates[idx] if timestamp - t <= 0.8]
            # print(coordinates)
            # if(idx==4):
            #     cv2.putText(image, f"thump",
            #                 (x, y), cv2.FONT_HERSHEY_DUPLEX,
            #                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return
def read_image(cap,hands):
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    results = hands.process(image)
    return image, results


def start():
    simulations_init()
    global RECOGNIZED
    global lock
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(
     model_complexity=1,
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5,
     max_num_hands = 1)
    while True:
        if time.time() - lock_time > duration_lock:
            lock = False
        image, results = read_image(cap,hands)
        if results.multi_hand_landmarks:
            draw_image(image,results)
            read_coordinates(image,results)
        operation(results)
        cv2.imshow("Test", image)
        time.sleep(0.02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    start()