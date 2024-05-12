import xlsxwriter
import numpy as np


# Create a new workbook
wb = xlsxwriter.Workbook('sim_block_big.xlsx')
sheet = wb.add_worksheet('Sheet1')	
sheet.write(0,0,"m")
sheet.write(0,1,"k")
sheet.write(0,2,"theta")
sheet.write(0,3,"mu")
sheet.write(0,4,"g")
sheet.write(0,5,"v_i")
sheet.write(0,6,"x_i")
sheet.write(0,7,"x_f")             
sheet.write(0,8,"pi1")
sheet.write(0,9,"pi2")
sheet.write(0,10,"pi3")
sheet.write(0,11,"pi4")
sheet.write(0,12,"pi5")
sheet.write(0,13,"pi6")
sheet.write(0,14,"pi7")


m  = 100.0
k = 200.0
g = 9.81
theta = 0.0
v_i = 0.0
x_i=0.0
dt = 0.0001
z = 0
                       
for j in range(1,4,1):
    if j == 1:
        mu = 0.2
    if j == 2:
        mu = 0.4
    if j == 3:
        mu = 0.9 
        
    for j in range(0,10,1):
        theta = theta+0.174533/2.0

        for j in range(0,50,1):
            v_i = v_i+0.1
            
            for j in range(0,10,1): 
                x_i = x_i+0.02
                
                x = x_i
                v = v_i
                while (v>0.0):
                   
                    a = (m*g*np.sin(theta)-(m*g*np.cos(theta)+k*x))/m
                    v = a*dt+v
                    x = v*dt+x
                    print(v)
                            
                z = z+1
                pi1 = k*x_i/m
                pi2 = g*x_i/v_i**2
                pi3 = theta
                pi4 = mu
                pi5 = x/x_i
                pi6 = (m*g*np.cos(theta)*mu+k*x_i)/(m*g*np.sin(theta))
                pi7 = (m*v_i**2)/(k*x_i**2)
                
                sheet.write(z,0,m)
                sheet.write(z,1,k)
                sheet.write(z,2,theta)
                sheet.write(z,3,mu)
                sheet.write(z,4,g)
                sheet.write(z,5,v_i)
                sheet.write(z,6,x_i)     
                sheet.write(z,7,x) 
                sheet.write(z,8,pi1)
                sheet.write(z,9,pi2)
                sheet.write(z,10,pi3)        
                sheet.write(z,11,pi4)        
                sheet.write(z,12,pi5)
                sheet.write(z,13,pi6) 
                sheet.write(z,14,pi7)

wb.close()

        
        
        



