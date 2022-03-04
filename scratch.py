def get_line_function(point1, point2):
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - a * point1[0]

    def line_function(x):
        return a * x + b

    return line_function


if __name__ == "__main__":
    l = get_line_function((1, 2), (3, 4))
    print(l(1))
    print(l(3))
    print(l(2))
