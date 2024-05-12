import xlsxwriter
import numpy as np
import random


# Create a new workbook
wb = xlsxwriter.Workbook('100kpendule.xlsx')
sheet = wb.add_worksheet('Sheet1')	
sheet.write(0,0,"m")
sheet.write(0,1,"l")
sheet.write(0,2,"g")
sheet.write(0,3,"v_i")
sheet.write(0,4,"h_i")
sheet.write(0,5,"v_f")
sheet.write(0,6,"pi1")             
sheet.write(0,7,"pi2")
sheet.write(0,8,"pi3")
sheet.write(0,9,"pi4")



g = 9.81
z = 0
                       

for j in range(0,100000,1):
    m = random.randrange(1,500,1)*0.01
    #m = 120.5
    for j in range(0,1,1):
        l = random.randrange(1,500,1)*0.01
        #l = 11.5
        for j in range(0,3,1):
            v_i = random.randrange(1,500,1)*0.01    
            for j in range(0,1,1): 
                h_i = random.randrange(1,1000,1)*0.001*l
                
                v_f = np.sqrt(2*(m*g*h_i+1/2*m*v_i**2)/m)
                print("v_f: "+str(v_f))
                print("h_i: "+str(h_i))
                print("v_i: "+str(v_i))
                print("m: "+str(m))
                print("l: "+str(l))
                
                z = z+1
                pi1 = v_f/v_i
                pi2 = g*l/v_i**2
                pi3 = h_i/l
                pi4 = (0.5*m*v_i**2)/(m*g*h_i)
                
                sheet.write(z,0,m)
                sheet.write(z,1,l)
                sheet.write(z,2,g)
                sheet.write(z,3,v_i)
                sheet.write(z,4,h_i)
                sheet.write(z,5,v_f)     
                sheet.write(z,6,pi1) 
                sheet.write(z,7,pi2)
                sheet.write(z,8,pi3)
                sheet.write(z,9,pi4)        


wb.close()

        
        
        



