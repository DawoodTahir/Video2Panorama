from math import sqrt
def quad():
    a=int(input('enter value of a : '))
    b=int(input('enter value of b : '))
    c=int(input('enter value of c : '))

    if a==0:
        raise ValueError('Its not a Quadratic Equation')
    elif b**2 - 4*a*c < 0:
        raise ValueError('No real Solution Exist')
    else:
        x1 = (-b + sqrt(b**2 - 4*a*c))/2*a
        x2 = (-b - sqrt(b**2 - 4*a*c))/2*a
    return x1,x2

print(quad())