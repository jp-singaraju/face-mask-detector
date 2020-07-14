# Do we iterate through a fixed number of times?
# Is it feasible to loop until the derivatives are actually zero?
from random import random


def a_der():
    der = 0
    for point in data:
        der += 2 * (a * float(point.split(",")[0]) + b - float(point.split(",")[1])) * float(point.split(",")[0])
    return der / len(data)


def b_der():
    der = 0
    for point in data:
        der += 2 * (a * float(point.split(",")[0]) + b - float(point.split(",")[1]))
    return der / len(data)


data = open(r"C:\Users\lavik\OneDrive\Desktop\Computer Science\Course and Camp Archives\UT Dallas Artificial Intelligence Workshop\Homework\Linear Regression\data.txt", "r").readlines()

a = 20 * random() - 10
b = 20 * random() - 10

for i in range(10000):
    new_a = a - ((1 / (1000000 * (i + 1))) * a_der())
    b -= (1 / (10000000000000000000000 * (i + 1))) * b_der()
    a = new_a

print(f"The equation of the line of best fit is y = {a}x + {b}.")
