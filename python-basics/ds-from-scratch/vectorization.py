# Vectorization and Matrices

def add_vectors(v, w) -> list:
    return [v_i + w_i for v_i, w_i in zip(v, w)];

def sub_vectors(v, w) -> list:
    return [v_i - w_i for v_i, w_i in zip(v, w)];

v = [4, 5, 2, 1];
w = [10, 12, 4, 5];

print(sub_vectors(w, v))