import xlsxwriter
import numpy as np


# Create a new workbook
wb = xlsxwriter.Workbook('civic.xlsx')
sheet = wb.add_worksheet('Sheet1')	
sheet.write(0,0,"v_i")
sheet.write(0,1,"l")
sheet.write(0,2,"a")
sheet.write(0,3,"delta")
sheet.write(0,4,"pi1")
sheet.write(0,5,"pi2")
sheet.write(0,6,"pi3")             
sheet.write(0,7,"pi4")
sheet.write(0,8,"pi5")
sheet.write(0,9,"X")
sheet.write(0,10,"Y")
sheet.write(0,11,"theta")
sheet.write(0,12,"pi6")

#Vehicle related params
l = 0.345

#To increment
z = 0

        
        
# Compute rear and front normal forces    
for v_i in range(1,51,1):
    v_i = v_i/0.1
    for a_u in range(1,11,1):
        a_u = a_u*0.1*9.81
        for delta in range(0,11,1):
            delta = float(delta)/10.0*45*np.pi/180
            z+=1
            v = v_i
            X = 0.0
            Y = 0.0
            theta = 0.0
            dt = 0.001
    
            while (v>0.0):
                print(v)
                X += v*np.cos(theta)*dt
                Y += v*np.sin(theta)*dt
                theta += v*np.tan(delta)*dt/l
                v += -a_u*dt
                

            pi1 = X/l
            pi2 = Y/l
            pi3 = theta
            pi4 = a_u*l/v_i**2
            pi5 = delta 
            pi6 = v_i**2*np.tan(delta)/(a_u*l)                  
                    
            sheet.write(z,0,v_i)
            sheet.write(z,1,l)
            sheet.write(z,2,a_u)
            sheet.write(z,3,delta)
            sheet.write(z,4,pi1)
            sheet.write(z,5,pi2)     
            sheet.write(z,6,pi3) 
            sheet.write(z,7,pi4)
            sheet.write(z,8,pi5)
            sheet.write(z,9,X)
            sheet.write(z,10,Y)
            sheet.write(z,11,theta)
            sheet.write(z,12,pi6)



wb.close()

        
        
        



