from turtle import *

def Recursive_Koch(length, depth):
    if depth == 0:
        forward(length)
    else:
        Recursive_Koch(length, depth-1)
        left(60)
        Recursive_Koch(length, depth-1)
        right(120)
        Recursive_Koch(length, depth-1)
        left(60)
        Recursive_Koch(length, depth-1)

# ----------
win=Screen()

penup()
backward(400)
pendown()

color('blue', 'green')
begin_fill()
speed(0)
Recursive_Koch(10, 3)
end_fill()

win.exitonclick()