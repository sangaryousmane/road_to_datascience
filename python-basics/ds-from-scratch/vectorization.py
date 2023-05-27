# Vectorization and Matrices
import math
from functools import reduce


def add_vectors(v, w) -> list:
    return [v_i + w_i for v_i, w_i in zip(v, w)];


def sub_vectors(v, w) -> list:
    return [v_i - w_i for v_i, w_i in zip(v, w)];


def vector_sum(vectors):
    return reduce(add_vectors, vectors)


def scalar_multiply(scalar, vector) -> list:
    return [scalar * v_1 for v_1 in vector];


# The dot product of two vectors is the sum of their componentwise products:
# the dot product is a scalar product that takes two vectors of the same dimension and produces a scalar value.
def dot(v, w) -> int:
    sum_ = 0
    lst = [v_i * w_i for v_i, w_i in zip(v, w)];
    for i in lst:
        sum_ += i;
    return sum_;


# v_i * v_i + v_2 * v_2 ..... + v_n * v_n
def sum_of_squares(v):
    return dot(v, v);


# Find the magnitude
def magnitude(v):
    return math.sqrt(sum_of_squares(v));


# (v_i - w_i)2 + (v_i - w_i)2 + ....
def square_distance(v, w):
    return sum_of_squares(sub_vectors(v, w));


def distance1(v, w):
    return math.sqrt(square_distance(v, w));


def distance2(v, w):
    return magnitude(sub_vectors(v, w));


v = [4, 5, 2, 1];
w = [10, 12, 4, 5];
# f = add_vectors(v, w)
print(f'Magnitude: {magnitude(v)}')
print(f'SD: {square_distance(v, w)}')
