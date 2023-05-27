from collections import Counter


def find_the_mode(l1):
    return Counter(l1);

def in_string(l: str) -> list:
    return Counter(l);

l = [3, 4, 2, 4, 1, 4, 2, 3]
l1 = "Hello brother";
c = Counter(l1);
print(c.most_common(2));

print(in_string(l1))
