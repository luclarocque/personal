import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# set up array of xvals on which to evaluate fcns 
xspace = np.linspace(0.001, 10, 6000)  

# y bounds
yub = 5
ylb = 1

# boundary functions
def hard_work(x=xspace):
	return 10**(1-x/8)

def peace(x=xspace):
	return 23**(1-(x+1.6)/4)

# don't have default xspace here -- must use l_xspace below
def anxiety(x):
	return 3**(0.5*(x-2.4))

def anx_hard_bd(x):
	return hard_work(x) - anxiety(x)

# we must limit the anxiety curve to prevent intersection with hard_work
# find x value where anxiety=hard_work 
xsol = fsolve(anx_hard_bd, 2)
# define new xspaces for this curve
l_xspace = np.linspace(0.001, float(xsol), 3000) # left side
r_xspace = np.linspace(float(xsol), 10, 3000) # right side


fig, ax = plt.subplots(1,1)

# plot boundaries
ax.plot(xspace, hard_work(), color='k')
ax.plot(xspace, peace(), color='k')
ax.plot(l_xspace, anxiety(l_xspace), color='k')

# set axis labels
plt.xlabel("Stress", fontsize=16)  # resistance to being (or effort)
plt.ylabel("Effort", fontsize=16)

# set axis boundaries
plt.xlim(1,6)
plt.ylim(ylb, yub)

# disable axis ticks (numbers)
plt.xticks([])
plt.yticks([])

# annotations
ax.annotate("Hard Work", xy=(4,4), xytext=(4,4), fontsize=20)
ax.annotate("Peace", xy=(1.1,1.3), xytext=(1.1,1.3), fontsize=16)
ax.annotate("Anxiety", xy=(4,1.5), xytext=(4,1.5), fontsize=20)
ax.annotate("Work", xy=(2.,3.), xytext=(2.,3.), fontsize=20)

# ---fill coloured regions---

# fill peace
ax.fill_between(xspace, ylb, peace(), 
	where=peace()>ylb, facecolor=(0.4,1,0.4), alpha=0.8)

# fill left half of anxiety (beneath anxiety curve)
# ax.fill_between(l_xspace, ylb, anxiety(l_xspace), 
# 	where=anxiety(r_xspace)>ylb, facecolor=(0.2,0.2,1), alpha=0.8)
ax.fill_between(xspace, ylb, anxiety(xspace), facecolor=(0.2,0.2,1), alpha=0.8)

# fill right half of anxiety (beneath hard work curve)
# ax.fill_between(r_xspace, ylb, hard_work(r_xspace), 
# 	where=hard_work(r_xspace)>ylb, facecolor=(0.2,0.2,1), alpha=0.8)

# fill work (keep alpha low since this overlaps anxiety)
ax.fill_between(xspace, peace(), hard_work(), 
	where=hard_work()>peace(), facecolor=(0.9,1.,0.), alpha=0.3)

# fill hard work (keep alpha=1 high since this must cover rest of anxiety curve)
ax.fill_between(xspace, hard_work(), yub, 
	where=yub>hard_work(), facecolor=(0.9,0.3,0.3), alpha=1)

plt.show()