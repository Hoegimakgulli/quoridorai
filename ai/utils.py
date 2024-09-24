import numpy as np


class Vector2:
    def __init__(self, x: int | float = None, y: int | float = None, list: list[int | float] = None, tuple: tuple[int | float] = None):
        if list is not None:
            self.x = list[0]
            self.y = list[1]
        elif tuple is not None:
            self.x = tuple[0]
            self.y = tuple[1]
        self.x = x if x is not None else 0
        self.y = y if y is not None else 0

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if type(other) != float and type(other) != int:
            raise TypeError("Multiplication is only supported with floats and integers")
        return Vector2(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vector2(self.x / other, self.y / other)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Index out of range")

    def __iter__(self):
        yield self.x
        yield self.y

    def length(self):
        return np.sqrt(self.x**2 + self.y**2)

    def normalize(self, inplace=False):
        if inplace:
            self /= self.length()
            return self
        return self / self.length()

    def distance(self, other):
        return (self - other).length()

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle(self, other):
        return np.arccos(self.dot(other) / (self.length() * other.length()))

    def rotate(self, angle):
        return Vector2(self.x * np.cos(angle) - self.y * np.sin(angle), self.x * np.sin(angle) + self.y * np.cos(angle))

    def to_tuple(self):
        return self.x, self.y

    def to_list(self):
        return [self.x, self.y]

    def zero():
        return Vector2(0, 0)

    def up():
        return Vector2(0, 1)

    def down():
        return Vector2(0, -1)

    def left():
        return Vector2(-1, 0)

    def right():
        return Vector2(1, 0)
