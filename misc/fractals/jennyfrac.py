from turtle import *

def Recursive_Koch(length, depth):
    if depth == 0:
        forward(length)
    else:
        Recursive_Koch(length, depth-1)
        left(45)
        Recursive_Koch(length, depth-1)
        right(45)
        Recursive_Koch(length/2, depth-1)
        right(45)
        Recursive_Koch(length, depth-1)
        left(45)
        Recursive_Koch(length, depth-1)


# ----------
win=Screen()

# change starting position
penup()
# left
backward(450)
# down
right(90)
forward(320)
left(90)
pendown()

color('purple', 'yellow')
begin_fill()
speed(0)
width(2)
Recursive_Koch(5, 4)
end_fill()

win.exitonclick()