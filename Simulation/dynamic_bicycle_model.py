import math
import numpy as np
import matplotlib.pyplot as plt

def dynamic_bicycle_model(L, vi, delta, a_u, mu, g, Nf, Nr, m, a , b, Iz):
    # Constants
    dt = 0.0001  # Time step

    # Tire parameters
    Fy_max_f = Nf*mu  # Maximum lateral force front tire (N)
    Fy_max_r = Nr*mu  # Maximum lateral force rear tire (N)
    #Fx_max_f = Nf*mu  # Maximum longitudinal force front tire (N) (Not used since front wheel rolls freely)
    Fx_max_r = Nr*mu  # Maximum longitudinal force rear tire (N)
    alpha_max_f = 0.1  # Maximum slip angle front tire (rad)
    alpha_max_r = 0.15  # Maximum slip angle rear tire (rad)
    kappa_max = 0.02  # Maximum slip ratio
    C_alpha_f = Fy_max_f/alpha_max_f  # Cornering stiffness front tire (N/rad)
    C_alpha_r = Fy_max_r/alpha_max_r  # Cornering stiffness rear tire (N/rad)
    #C_x_f = Fx_max_f/kappa_max  # Longitudinal stiffness front tire (N) (Not used since front wheel rolls freely)
    C_x_r = Fx_max_r/kappa_max  # Longitudinal stiffness rear tire (N)

    
    # Initial conditions
    x = 0.0
    y = 0.0
    theta = 0.0
    v_x = vi
    omegaR = vi
    v_y = 0.0
    dtheta = 0.0
    t = 0.0
    Sx_r = 0.0
    alpha_f = 0.0
    alpha_r = 0.0
    Fx_r = 0.0
    Fy_r = 0.0
    Fy_f = 0.0
    X = []
    Y = []
    T = []
    time = []
    Sx_r_list = []
    alpha_r_list = []
    alpha_f_list = []
    Fx_r_list = []
    Fy_r_list = []
    Fy_f_list = []
    o_list = []
    v_list = [] 
    
    while v_x > 0:
       
        # Update slip angles for front and rear wheels
        alpha_r = -np.arctan((v_y-b*dtheta)/v_x)
        alpha_f = delta-np.arctan((v_y+a*dtheta)/v_x)
        
        # Compute rear wheel deceleration according to chosen manoeuvre
        omegaR = omegaR-a_u*dt

                               
        # Update slip ratio for rear wheel (Sx_f = 0 since front wheel rolls freely)
        if v_x <= 0.0:
            Sx_r = 0.0
        else:
            Sx_r = (omegaR-v_x)/v_x
        
        # Compute Fy_f in linear region and in plateau region of tire model and keep de minimum
        Fy_f = C_alpha_f * alpha_f
        if Fy_f < 0:
            Fy_f = -min(abs(Fy_f), Fy_max_f)
        else:
            Fy_f = min(Fy_f, Fy_max_f)
            
        # Compute Fy_r in linear region and in plateau region of tire model and keep de minimum
        Fy_r = C_alpha_r * alpha_r
        if Fy_r < 0:
            Fy_r = -min(abs(Fy_r), Fy_max_r)
        else:
            Fy_r = min(Fy_r, Fy_max_r)
        
        # Compute Fx_f (front wheel rolls freely)
        #Fx_f = 0.0
        
        # Compute Fx_r in linear region and in plateau region of tire model and keep de minimum
        Fx_r = C_x_r * Sx_r
        if Fx_r < 0:
            Fx_r = -min(abs(Fx_r), Fx_max_r)
        else:
            Fx_r = min(Fx_r, Fx_max_r)
            
        #Combined slip for front wheel (only latteral force)
        #F_f = Fy_f
        #F_f_ang = np.pi/2-delta #90 degrees according to front wheel 
        
        #Combined slip for rear wheel maxed out at Nr*mu
        #F_r = np.sqrt(Fx_r**2+Fy_r**2)
        #F_r = min(F_r,Nr*mu)
        #F_r_ang = math.atan2(Fy_r, Fx_r)
        
        v_x +=  ((-Fy_f*math.sin(delta)+Fx_r)/m+v_y*dtheta)*dt
        if v_x < 0.05:
            v_x = 0.0
        print(v_x)
        v_y +=  ((Fy_f*math.cos(delta)+Fy_r)/m - v_x*dtheta)*dt
        dtheta += ((a*(Fy_f*math.cos(delta))-b*Fy_r)/Iz)*dt

        
        # Sum forces and moments
        #v_x += ((F_r*math.cos(F_r_ang)-F_f*math.cos(F_f_ang))/m+v_y*dtheta)*dt
        #if v_x < 0.05:
        #    v_x = 0.0
        #v_y += ((F_r*math.sin(F_r_ang)+F_f*math.sin(F_f_ang))/m-v_x*dtheta)*dt
        #dtheta += (a*F_f*math.sin(F_f_ang)-b*F_r*math.sin(F_r_ang))/Iz*dt
        #print(v_x)
        
        # Update position and orientation
        x += (v_x*math.cos(theta)-v_y*math.sin(theta))*dt
        y += (v_x*math.sin(theta)+v_y*math.cos(theta))*dt
        theta += dtheta*dt
        
        X.append(x)
        Y.append(y)
        T.append(theta)
        time.append(t)
        Sx_r_list.append(Sx_r)
        alpha_r_list.append(alpha_r)
        alpha_f_list.append(alpha_f)
        Fx_r_list.append(Fx_r)
        Fy_r_list.append(Fy_r)
        Fy_f_list.append(Fy_f)
        o_list.append(omegaR)
        v_list.append(v_x)
                
        # Update time
        t += dt
        
    return x, y, theta, X, Y, T, time, Sx_r_list, alpha_f_list, alpha_r_list, Fx_r_list, Fy_r_list, Fy_f_list, o_list, v_list

# Input parameters
g = 9.81                            # Gravitational acceleration (m/s^2)
L = 1.2                             # Wheelbase
vi = 5.0                            # Initial speed (m/s)
delta = math.radians(35)             # Steering angle in radians
a_u = 0.2*g                         # Backwheel deceleration (rad/s^2)
mu = 0.8                            # Friction coefficient
cg = 1.31                           # Mass center
m = 6.79                            # Mass (kg)
Nr = m/(cg+1)*9.81                  # Normal force on rear wheels (N)
Nf = m*9.81-Nr                      # Normal force on front wheels (N)
b = Nf/(m*9.81)*L                   # Length from cg to rear wheels (m)
a = L-b                             # Length from cg to front wheels (m)
w = 0.4                             # Width (m)
Iz = 1.00/12.00*m*(w**2+L**2)       # Inertia from rotation of the yaw

# Calls for model
x, y, theta, X, Y, T, time, Sx_r_list, alpha_f_list, alpha_r_list, Fx_r_list, Fy_r_list, Fy_f_list, o_list, v_list = dynamic_bicycle_model(L, vi, delta, a_u, mu, g, Nf, Nr, m, a , b, Iz)

print("Final position: ({:.2f}, {:.2f})".format(x, y))
print("Final orientation: {:.2f} radians".format(theta))


plt.figure(1)
plt.plot(X,Y,'--r', label='SIMULATION ')
plt.arrow(x,y,0.05*math.cos(theta),0.05*math.sin(theta), head_width=0.025, head_length=0.05)
plt.title("Trajectoire SIMULATION \n Decel: "+str(a_u)+"m/s**2, Steering: "+str(delta)+"rad , Vitesse: "+str(vi)+", Mu: "+str(mu))
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

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)

fig.suptitle("Slip")

ax1.plot(time, Sx_r_list,  '--r',label="Sim")
ax1.set_ylabel('Slip ratio rear wheel')
ax2.plot(time, alpha_f_list,  '--g',label="Sim")
ax2.set_ylabel('Slip angle front wheel (rad)')
ax3.plot(time,alpha_r_list,  '--k',label="Sim")
ax3.set_ylabel('Slip angle rear wheel (rad)')
ax3.set_xlabel('Time (s)')
ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)

fig.suptitle("Forces")

ax1.plot(time, Fx_r_list,  '--r',label="Sim")
ax1.set_ylabel('Longitudinal force on rear wheel (N)')
ax2.plot(time, Fy_r_list,  '--g',label="Sim")
ax2.set_ylabel('Latteral force on rear wheel (N)')
ax3.plot(time,Fy_f_list,  '--k',label="Sim")
ax3.set_ylabel('Latteral force on front wheel (N)')
ax3.set_xlabel('Time (s)')
ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)

plt.figure(5)
plt.plot(time,o_list,'--r', label='OmegaR ')
plt.plot(time,v_list,'--g', label='v_x ')
plt.title("Vitesses")
plt.xlabel('Time (s)')
plt.ylabel('Vitesses (m/s)')
plt.axis("equal")
plt.legend()

plt.show()
