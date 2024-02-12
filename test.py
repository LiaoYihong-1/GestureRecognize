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
from rule import Key_Type, keys_down
coordinates = [[] for i in range(21)]
# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
RECOGNIZED = False
TRACE_TIME = 0.4
LEFT = "L"
DOWN = "D"
RIGHT = "R"
UP = "U"
MID = "M"
FRONT = "F"
BACK = "B"
SLEEP_TIME = 0.005
simulation_x = None
simulation_y = None
simulation_z = None
definitions_move_x = []
definitions_move_y = []
definitions_move_z = []
hands = mp.solutions.hands.Hands(
     model_complexity=1,
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5,
     max_num_hands = 1)
lock = False
lock_time = 0
in_continuous_operation = False
continuous_operation_time = None
continuous_operation = ''
continuous_page_next = "CPN"
def calculate_pluse(x1, x2)->float:
    if x2-x1>10:
        return 10
    elif x2-x1<-10:
        return -10
    return x2-x1

def clear():
    global coordinates
    coordinates = [[] for i in range(21)]


def result_generate(result:[])->str:
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
    k_values[B] = fuzz.trimf(k_values.universe, [-4, 0, 4])
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
    # movement_state.view(sim=simulation)
    return simulation

def simulations_init():
    global simulation_x
    global simulation_y
    global simulation_z
    simulation_x = simulation_init(LEFT,MID,RIGHT)
    simulation_y = simulation_init(UP,MID,DOWN)
    simulation_z = simulation_init(FRONT,MID,BACK)


def separate_coordinates()->tuple[list[list], list[list], list[list]]:
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
    return x_labels, y_labels, z_labels

def map_output_to_state(output_value, axis='x')->str:
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

def fuzzy()->[]:
    global simulation_x
    global definitions_move_x
    global simulation_y
    global definitions_move_y
    global simulation_z
    global definitions_move_z
    global coordinates
    x_labels,y_labels,z_labels = separate_coordinates()
    x_result = []
    y_result = []
    z_result = []
    figures_gaps_x = []
    figures_gaps_y = []
    figures_gaps_z = []
    for i in range(len(x_labels)):
        figure_gap_x = []
        figure_gap_y = []
        figure_gap_z = []
        for l in range(len(x_labels[i])-1):
            g_x = calculate_pluse(x_labels[i][l],x_labels[i][l+1])
            g_y = calculate_pluse(y_labels[i][l],y_labels[i][l+1])
            g_z = calculate_pluse(z_labels[i][l],z_labels[i][l+1])
            figure_gap_x.append(g_x)
            figure_gap_y.append(g_y)
            figure_gap_z.append(g_z)
        figures_gaps_x.append(figure_gap_x)
        figures_gaps_y.append(figure_gap_y)
        figures_gaps_z.append(figure_gap_z)
    for i in figures_gaps_x:
        result = []
        for k in i:
            simulation_x.input["k"] = k
            simulation_x.compute()
            result.append(map_output_to_state(simulation_x.output['move'],'x'))
        x_result.append(result)
    for i in figures_gaps_y:
        result = []
        for k in i:
            simulation_y.input["k"] = k
            simulation_y.compute()
            result.append(map_output_to_state(simulation_y.output['move'],'y'))
        y_result.append(result)
    for i in figures_gaps_z:
        result = []
        for k in i:
            simulation_z.input["k"] = k
            simulation_z.compute()
            result.append(map_output_to_state(simulation_z.output['move'],'z'))
        z_result.append(result)
    return x_result, y_result, z_result

def read_move():
    global definitions_move_x
    global definitions_move_y
    global definitions_move_z
    x_movement, y_movement, z_movement = fuzzy()
    definitions_move_x = [result_generate(m) for m in x_movement]
    definitions_move_y = [result_generate(m) for m in y_movement]
    definitions_move_z = [result_generate(m) for m in z_movement]


def is_page_next()->bool:
    # return definitions_move_x[0] == MID and definitions_move_y[8] == MID and definitions_move_y[12] == MID and definitions_move_y[16] == MID \
    #  and definitions_move_x[8] == RIGHT and \
    #             definitions_move_x[12] == RIGHT and definitions_move_x[16] == RIGHT
    return dont_move(4) and dont_move(20) and dont_move(16) and dont_move(0) \
            and definitions_move_y[8] == DOWN and definitions_move_y[12] == DOWN

def is_page_up()->bool:
    return definitions_move_x[0] == MID and definitions_move_y[8] == MID and definitions_move_y[8] == MID and definitions_move_y[12] == MID and definitions_move_y[16] == MID \
           and definitions_move_x[8] == LEFT and definitions_move_x[12] == LEFT and definitions_move_x[16] == LEFT

def is_desktop()->bool:
    return dont_move(0) and definitions_move_y[8] == DOWN and definitions_move_y[12] == DOWN and \
                definitions_move_y[16] == DOWN and definitions_move_x[20] == MID

def dont_move(figure: int)->bool:
    return definitions_move_x[figure] == MID and definitions_move_y[figure] == MID

def is_continuous_page_next()->bool:
    return dont_move(5) and dont_move(9) and dont_move(13) and dont_move(17) and dont_move(8) and dont_move(12) and dont_move(16) \
           and definitions_move_y[4] == DOWN

def is_continuous_page_last()->bool:
    return dont_move(5) and dont_move(9) and dont_move(13) and dont_move(17) and dont_move(4) and dont_move(12) and dont_move(16) \
           and definitions_move_x[8] == LEFT

def destroy():
    global in_continuous_operation
    clear()
    in_continuous_operation = False
    for i in keys_down.keys():
        key_up(i)

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


def operation(results):
    global RECOGNIZED
    global coordinates
    global TRACE_TIME
    global lock
    global lock_time
    global continuous_operation_time
    global in_continuous_operation
    global continuous_operation
    if time.time() - lock_time > 1:
        lock = False
    if results.multi_hand_landmarks:
        if not RECOGNIZED:
            RECOGNIZED = True
        if lock:
            return
        if time.time() - coordinates[0][0][3] < TRACE_TIME * 2 / 3:
            return
        read_move()
        hand = results.multi_handedness[0].classification[0].label
        if in_continuous_operation:
            if continuous_operation == continuous_page_next:
                if time.time() - continuous_operation_time > 2:
                    key_up(Key_Type.TAB)
                    key_up(Key_Type.ALT)
                    in_continuous_operation = False
                    continuous_operation = None
                else:
                    if is_continuous_page_next():
                        press(Key_Type.RIGHT)
                        clear()
                        print("page next con")
                        continuous_operation_time = time.time()
                    elif is_continuous_page_last():
                        press(Key_Type.LEFT)
                        clear()
                        print("page last con")
                        continuous_operation_time = time.time()
        else:
            if hand == 'Left':
                if is_page_next():
                    hot_key(Key_Type.ALT, Key_Type.TAB)
                    lock = True
                    lock_time = time.time()
                    clear()
                    print("page next")
                elif is_continuous_page_next():
                    key_down(Key_Type.ALT)
                    key_down(Key_Type.TAB)
                    press(Key_Type.RIGHT)
                    clear()
                    print("page next con")
                    in_continuous_operation = True
                    continuous_operation_time = time.time()
                    continuous_operation = continuous_page_next
            elif hand == 'Right':
                if is_page_up():
                    key_down(Key_Type.ALT)
                    key_down(Key_Type.TAB)
                    press(Key_Type.LEFT)
                    press(Key_Type.LEFT)
                    key_down(Key_Type.TAB)
                    key_down(Key_Type.ALT)
                    lock = True
                    lock_time = time.time()
                    clear()
                    print("page up")
            elif is_desktop():
                pa.hotkey(Key_Type.WIN.value, Key_Type.D.value)
                lock = True
                lock_time = time.time()
                clear()
                print("desktop")
        test([4],definitions_move_y)
        ## print(definition_move)
    else:
        if RECOGNIZED:
            destroy()
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
            coordinates[idx] = [(x, y, z, t) for x, y, z, t in coordinates[idx] if timestamp - t <= TRACE_TIME]
            ## print(coordinates)
            # if(idx==4):
            #     cv2.putText(image, f"thump",
            #                 (x, y), cv2.FONT_HERSHEY_DUPLEX,
            #                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return

def process_image(image):
    image = cv2.flip(image, 1)
    results = hands.process(image)
    return image,results

def start():
    simulations_init()
    global RECOGNIZED
    global lock
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        image,results = process_image(image)
        if results.multi_hand_landmarks:
            draw_image(image,results)
            read_coordinates(image,results)
        operation(results)
        cv2.imshow("Test", image)
        time.sleep(SLEEP_TIME)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    start()
    destroy()