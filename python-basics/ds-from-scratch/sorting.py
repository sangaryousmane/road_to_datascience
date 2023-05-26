# Sorting in python
import random as rd
import re

nums = [3, 4, 10, 11, 2, 13, -1]


def sort_them():
    num = [i for i in nums if i % 2 == 0]
    sorted(nums, reverse=True)


def sort_them2() -> list:
    result = [(i, j) for i in range(10) for j in range(20)]
    return result;


def sort_them3() -> list:
    dict_list = {x: x * x for x in range(10)}
    return dict_list;


def generator_1():
    n = 0
    while True:
        yield n;
        n += 1;

def randomness():
    # rn = rd.randrange(1, 10)
    lst = range(10)
    # rn = rd.sample(lst, 3)
    # rn = rd.uniform(5, 0)
    rn = rd.choice(nums)
    print(rn)

def regular_exp():
    print(re.search("o", "create"));


def display1():
    for i in generator_1():
        print(i)
        if i >= 10:
            break;


regular_exp();

