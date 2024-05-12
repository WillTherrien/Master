import math
import numpy as np
import matplotlib.pyplot as plt

def magic_formula_longitudinal(C, D, E, B, Sx, N):
    Fz = N  # Vertical load on the tire (N)

    S = np.sin(C * np.arctan(B * Sx - E * (B * Sx - np.arctan(B * Sx))))
    Fx = D * Fz * S

    return Fx

def magic_formula_lateral(C, B, alphas, N):
    Fz = N  # Vertical load on the tire (N)

    S = np.sin(C * np.arctan(B * alphas))
    Fy = Fz * S

    return Fy

def dynamic_bicycle_model(L, vi, delta_target, a_u, mu, g, Nf, Nr, m, a , b, Iz):
    # Constants
    dt = 0.0001  # Time step
    
    

    # Tire parameters
    Fy_max_f = Nf*mu  # Maximum lateral force front tire (N)
    Fy_max_r = Nr*mu  # Maximum lateral force rear tire (N)
    #Fx_max_f = Nf*mu  # Maximum longitudinal force front tire (N) (Not used since front wheel rolls freely)
    Fx_max_r = Nr*mu  # Maximum longitudinal force rear tire (N)
    alpha_max_f = 0.1  # Maximum slip angle front tire (rad)
    alpha_max_r = 0.1  # Maximum slip angle rear tire (rad)
    kappa_max = 0.02  # Maximum slip ratio


    
    # Initial conditions
    x = 0.0
    y = 0.0
    theta = 0.0
    delta = 0.0
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
    
    # Example usage
    C_long = 1.2  # Longitudinal stiffness
    D_long = 1.0  # Longitudinal shape factor
    E_long = 0.8  # Longitudinal curvature factor
    B_long = 10.0  # Longitudinal slip stiffness

    C_lat = 1.2  # Lateral stiffness
    B_lat = 10.0  # Lateral slip stiffness
    
    while v_x > 0:
        
        delta += 0.001
        delta = min(delta,delta_target)
        
        # Compute rear wheel deceleration according to chosen manoeuvre
        omegaR = max(omegaR-a_u*dt,0.0)
       
        
        #Tire longitudinal speed
        vx_f = v_x*np.cos(delta)+(v_y+a*dtheta)*np.sin(delta)
        vx_r = v_x
        
        #Longitudinal slip ratio
        Sx_r = (omegaR-vx_r)/vx_r
        
        #Slip angles
        alpha_f = (delta-np.arctan((v_y+a*dtheta)/vx_f))
        alpha_r = np.arctan((b*dtheta-v_y)/vx_r)
        
        # Compute forces using Magic Formula
        Fx_r = magic_formula_longitudinal(C_long, D_long, E_long, B_long, Sx_r, Nr)
        Fy_r = magic_formula_lateral(C_lat, B_lat, alpha_r, Nr)
        Fy_f = magic_formula_lateral(C_lat, B_lat, alpha_f, Nf)

            
        # Sum forces and moments
        v_x +=  ((Fx_r-Fy_f*np.sin(delta))/m+v_y*dtheta)*dt
        v_y +=  ((Fy_f*np.cos(delta)+Fy_r)/m - v_x*dtheta)*dt
        dtheta += ((a*(Fy_f*np.cos(delta))-b*Fy_r)/Iz)*dt
        if v_x<0.01:
            v_x=0.0
        print(v_x)
        
        # Update position and orientation
        x += (v_x*np.cos(theta)-v_y*np.sin(theta))*dt
        y += (v_x*np.sin(theta)+v_y*np.cos(theta))*dt
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
vi = 2.0                            # Initial speed (m/s)
delta_target = math.radians(40)             # Steering angle in radians
a_u = 0.2*g                         # Backwheel deceleration (rad/s^2)
mu = 0.2                            # Friction coefficient
cg = 1.31                           # Mass center
m = 6.79                            # Mass (kg)
Nr = m/(cg+1)*9.81                  # Normal force on rear wheels (N)
Nf = m*9.81-Nr                      # Normal force on front wheels (N)
b = Nf/(m*9.81)*L                   # Length from cg to rear wheels (m)
a = L-b                             # Length from cg to front wheels (m)
w = 0.4                             # Width (m)
Iz = 1.00/12.00*m*(w**2+L**2)       # Inertia from rotation of the yaw



# Calls for model
x, y, theta, X, Y, T, time, Sx_r_list, alpha_f_list, alpha_r_list, Fx_r_list, Fy_r_list, Fy_f_list, o_list, v_list = dynamic_bicycle_model(L, vi, delta_target, a_u, mu, g, Nf, Nr, m, a , b, Iz)

print("Final position: ({:.2f}, {:.2f})".format(x, y))
print("Final orientation: {:.2f} radians".format(theta))


plt.figure(1)
plt.plot(X,Y,'--r', label='SIMULATION ')
plt.arrow(x,y,0.05*np.cos(theta),0.05*np.sin(theta), head_width=0.025, head_length=0.05)
plt.title("Trajectoire SIMULATION \n Decel: "+str(a_u)+"m/s**2, Steering: "+str(delta_target)+"rad , Vitesse: "+str(vi)+", Mu: "+str(mu))
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
