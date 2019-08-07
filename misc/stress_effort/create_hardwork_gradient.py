import imageio
import matplotlib.pyplot as plt
import numpy as np

# Image size
width = 640
height = 480
channels = 3

# Create an empty image
img = np.zeros((height, width, channels), dtype=np.uint8)

# Set up pixel grid
xs = np.linspace(0, width, num=width, endpoint=False)
ys = np.linspace(0, height, num=height, endpoint=False)
yy, xx = np.mgrid[:height,:width]

# Draw peace
scale = 0.95
pcircle = np.power( ((yy - height)/height)**2 + ((xx - 0)/width)**2, 1/2.)*300/scale  # define circle for peace
peace = np.clip(pcircle , a_min=0, a_max=255)  # ensure all values between 0-255
peace = 255 - peace

# Draw work
scale = 1.1
wcircle = np.power( ((yy - 250)/height)**4 + ((xx + 200)/width)**4, 1/4.)*300/scale
work = np.clip(wcircle , a_min=0, a_max=255)
work = 255 - work

# Draw anxiety
scale = 1
acircle = np.power( ((yy - height-150)/height)**2 + ((xx - width+400)/width)**2 , 1/2.)*300/scale
anx = np.clip(acircle , a_min=0, a_max=255)
anx = 255 - anx

# Set the RGB values
for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        r, g, b = (	anx[x][y],
                    peace[x][y],
                	work[x][y])
        img[x][y][0] = r
        img[x][y][1] = g
        img[x][y][2] = b

# set axis labels
plt.xlabel("Stress", fontsize=16)  # resistance to being (or effort)
plt.ylabel("Effort", fontsize=16)

# disable axis ticks (numbers)
plt.xticks([])
plt.yticks([])

# annotations
plt.annotate("Hard\nWork", xy=(500,100), xytext=(500,100), fontsize=20, color='white')
plt.annotate("Peace", xy=(50,450), xytext=(50,450), fontsize=20, color='white')
plt.annotate("Anxiety", xy=(400,450), xytext=(400,450), fontsize=20, color='white')
plt.annotate("Work", xy=(160,160), xytext=(160,160), fontsize=20, color='white')

# Add image to figure
plt.imshow(img)

# Save the image
plt.savefig("stress_effort_gradient.png")

# Display the image
plt.show()

