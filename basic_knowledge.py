from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("Experience", ['state', 'action', 'reward', 'done'])


class A:
    def __init__(self, a, b, c):
        self.a = a          
        self.b = b
        self.c = c

    def __call__(self, x, y):
        self.x = x
        self.y = y
        print("in __call__(): ", self.x, self.y)

    def __iter__(self):
        histories = []
        for _ in range(5):
            histories.append(deque(maxlen=10))

        states = [[1,2,3,4,5], [22,3,44,55,90]]
        action = [[1,1,1,1,1,1], [0,6,4,2,1,1]]
        reward = [[44], [90]]
        done = [[True], [False]]
        for i in range(2):
            history = histories[i]
            history.append(Experience(state=states, action=action, reward=reward, done=done))
            yield tuple(history)
            

a = A(1, 2, 3)
a(4, 5)
# for i in a:
#     print(i[0].state)
print(a[0])