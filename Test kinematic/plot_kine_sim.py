import math
import numpy as np
import matplotlib.pyplot as plt

def kinematic_bicycle_model(l, vi, delta, a_u):
    # Constants
    dt = 0.0001  # Time step
    
    # Initial conditions
    x = 0.0
    y = 0.0
    theta = 0.0
    v = vi
    t = 0.0
    X = []
    Y = []
    T = []
    time = []
    v_list = [] 
    
    while v > 0:

	X.append(x)
        Y.append(y)
        T.append(theta)
        time.append(t)
        v_list.append(v)
        
        x += v*np.cos(theta)*dt
        y += v*np.sin(theta)*dt
        theta += v*np.tan(delta)*dt/l
        v += -a_u*dt
                
        # Update time
        t += dt
        
    return x, y, theta, X, Y, T, time, v_list

# Input parameters
l = 1.2                             # Wheelbase
vi = 10.0                            # Initial speed (m/s)
delta = math.radians(40)             # Steering angle in radians
g =9.81
a_u = 0.9*g                         # Backwheel deceleration (rad/s^2)

# Calls for model
x, y, theta, X, Y, T, time, v_list = kinematic_bicycle_model(l, vi, delta, a_u)

print("Final position: ({:.2f}, {:.2f})".format(x, y))
print("Final orientation: {:.2f} radians".format(theta))


plt.figure(1)
plt.plot(X,Y,'--r', label='SIMULATION ')
plt.arrow(x,y,0.05*np.cos(theta),0.05*np.sin(theta), head_width=0.025, head_length=0.05)
plt.title("Trajectoire SIMULATION")
plt.xlabel('X_position (m)')
plt.ylabel('Y_position (m)')
plt.axis("equal")
plt.legend()

plt.figure(2)
plt.plot(time,T,'--b', label='SIMULATION ')
plt.title("Orientation dans le temps")
plt.xlabel('Temps (s)')
plt.ylabel('Orientation (rad)')
plt.axis("equal")
plt.legend()

plt.figure(3)
plt.plot(time,v_list,'--g', label='v')
plt.title("Vitesses")
plt.xlabel('Time (s)')
plt.ylabel('Vitesses (m/s)')
plt.axis("equal")
plt.legend()

plt.show()
