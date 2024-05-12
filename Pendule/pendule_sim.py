import xlsxwriter
import numpy as np


# Create a new workbook
wb = xlsxwriter.Workbook('pendule_1.xlsx')
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


m  = 5.0
l = 1.0
g = 9.81
v_i = 0.0
h_i=l
dt = 0.0001
z = 0
                       


for j in range(0,500,1):
    v_i = v_i+0.01
    
    
    h_i=l
    for j in range(0,100,1): 
        h_i = h_i-l/110.0
        print("h_i: "+str(h_i))
        
        v_f = np.sqrt(2*(m*g*h_i+1/2*m*v_i**2)/m)
        #print("v_f: "+str(v_f))
                    
        z = z+1
        pi1 = v_f/v_i
        pi2 = g*l/v_i**2
        pi3 = h_i/l
	print("v_i: "+str(v_i))
	print("1/2mv2: "+str(0.5*m*v_i**2))
	print("m: "+str(m))
	print("mgh: "+str(m*g*h_i))
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

        
        
        



