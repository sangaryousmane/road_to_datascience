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
    print(f' a = {a}, b = {b}, args = {args}, optin = {option}')
    print(f'kwargs = {kwargs}')


# pack and unpacking
def args_pack(*args):
    print(args);

def kwargs_unpack(**kwargs):
    print(kwargs);

def zipping_list(lst1:list, lst2:list):
    for i1, i2 in zip(lst1, lst2):
        print(f'{i1, i2}')

def nationalities(l1, l2):

    for name, country in zip(l1, l2):
        print(f'{name} is from {country}')

lst1 = ["Ousmane", "Puyol", "Thomas", "Zeynab"]
lst2 = ["Liberia", "DR Congo", "Gabon", "Morocco"]

nationalities(lst1, lst2)
# lst = [2, 4, 5]
# dicts = {"age": 23, "name": "Ousmane"}
#
# kwargs_unpack(country="Liberia", hobby="Coding", **dicts);
# args_pack(*lst);
# args_kwargs(4, 5, 9, 10, 11, option=True, option1=3, option2=5);
# arg_printer(4, 5, True, param=1, param1=2, param2=3, param3=4);
# print(f' Sum is: {addition(4, 5, 2, 5, 6, 1, 3, 10, option=True)}');
