import math

def linear(a, b, x):
    return a*x+b


def quad(a, b, c, x):
    return a*x*x+b*x+c


def sin_f(a, w, f, x):
    return a*math.sin(w*x + f)


def cos_f(a, w, f, x):
    return a*math.cos(w*x + f)


def root_n(x, n=3):
    if n%2 == 1:
        return x ** (1./n)
    elif n%2 == 0 and x > 0:
        return x ** (1./n)


def log_f(x, base=math.e):
    if x > 0:
        return math.log(x, base)