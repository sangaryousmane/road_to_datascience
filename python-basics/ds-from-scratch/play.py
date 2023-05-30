# Zip pack, unpacking,  *args and **kwargs

def sum(*args):
    total = 0
    for i in args:
        total += i;
    return total;


# (a, b) are positional arguments. c is a keyword argument
def manual(a, b, c=10, *args):
    print(f'{a}, {b}, {c}, the rest are: {args}')


def addition(a, b, *args, option=True):
    sum = 0;

    if option:
        for i in args:
            sum += i;
        return a + b + sum;
    else:
        return sum;


def arg_printer(a, b, option, **kwargs):
    print(f'a = {a}, b = {b}')
    print(f'{option}')
    print(f'{kwargs}')


def args_kwargs(a, b, *args, option=True, **kwargs):
    print(f' a = {a}, b = {b}, args = {args}, option = {option}')
    print(f'kwargs = {kwargs}')


# pack and unpacking
def args_pack(*args):
    print(args);


# Accessing all positional arguments
def kwargs_unpack(**kwargs):
    print(kwargs);


def zipping_list(lst1: list, lst2: list):
    for i1, i2 in zip(lst1, lst2):
        print(f'{i1, i2}')


def nationalities(l1, l2):
    for name, country in zip(l1, l2):
        print(f'{name} is from {country}')


lst1 = ["Ousmane", "Puyol", "Thomas", "Zeynab"]
lst2 = ["Liberia", "DR Congo", "Gabon", "Morocco"]

# nationalities(lst1, lst2)
# lst = [2, 4, 5]
# dicts = {"age": 23, "name": "Ousmane"}
#
# kwargs_unpack(country="Liberia", hobby="Coding", **dicts);
# args_pack(*lst);
# args_kwargs(4, 5, 9, 10, 11, option=True, option1=3, option2=5);
# arg_printer(4, 5, True, param=1, param1=2, param2=3, param3=4);
# print(f' Sum is: {addition(4, 5, 2, 5, 6, 1, 3, 10, option=True)}');
import numpy as np
# Matplotlib series
from matplotlib import pyplot as plt

ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000,
            71496, 75370, 83640, 84666,
            84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708,
            108423, 101407, 112542, 122870, 120000]
dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928,
         67317,
         68748, 73752, 77232,
         78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988,
         100000, 108923, 105000, 103117]
x_indexes = np.arange(len(ages_x))
width = .25


def line_trend():
    # Dark background
    plt.style.use('fivethirtyeight')
    # Use the xkcd styling
    # plt.xkcd()
    # Title
    plt.title("Stack Overflow Medium salary by age")
    # Plot the two
    plt.plot(x_indexes - width, dev_y, label='All Dev')
    plt.plot(x_indexes + width, py_dev_y, label='Python Dev')
    plt.xlabel("Salary $. OUSMANE SANGARY")
    plt.ylabel("Ages.")
    # Save figure
    plt.savefig("dev_pay.png")
    plt.legend(loc=9)
    plt.tight_layout()
    plt.figure(figsize=(17, 7))
    plt.show()


def bar_trend():
    plt.style.use('fivethirtyeight')
    plt.bar(x_indexes + width, py_dev_y, width=width, label='Python devs')
    plt.bar(x_indexes - width, dev_y, width=width, label='All Dev')

    plt.xlabel("Salary $")
    plt.ylabel("Ages.")
    plt.legend(loc=9)
    plt.show()


def palindrome(word):
    return word == word[::-1];


def sumOfNumbers(n):
    i: int = 0;
    if n < 0:
        n *= -1;

    if n == 0:
        return 1;

    while n != 0:
        i +=1;
        n //= 10;
    return i;


print(f'{sumOfNumbers(-3353320)}');
