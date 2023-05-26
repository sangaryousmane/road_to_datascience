# Mimics the Set class

class Set:
    def __init__(self, values=None):
        self.dict = {}

        if values is not None:
            for value in values:
                self.add(value);

    # Return the string representation of the list
    def __repr__(self):
        return "Set: " + str(self.dict.keys())

    def add(self, value):
        self.dict[value] = True;

    # check for a value in a set
    def contains(self, value):
        return value in self.dict;

    # Remove a value in a set
    def remove(self, value):
        if value in self.dict:
            del self.dict[value];
        else:
            raise KeyError("Sorry, that value isn't in the set")


myAccess = Set([1, 4, 2, 10, 8]);
myAccess.add(11)
myAccess.remove(1)
print(myAccess)
# print(myAccess.contains(3))
