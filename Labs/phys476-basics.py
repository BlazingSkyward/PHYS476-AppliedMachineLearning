""" This is actually a python 2.7 file, but everything here will work in python
3, including the printing, which was done with the python 3 print_function
import below. 

The numpy and/or scipy libraries are almost automatically required for any
scientific coding. 

Note the two different kinds of commenting... this type, for major notes and
titles, vs # for shorter aside comments and commenting out. """

from __future__ import print_function
import numpy as np

""" basics """
#print('Hello world')

#a = 5.2343
#b = 6.4543
#prod = a*b
#exp = a**b   # exponentiation with **, rather than ^
#x = np.sin(b)  # many basic functions are found in the numpy library

""" vector and matrix basics. """
#vec = np.array([2.,3.,4.])
#vec5 = np.array([2.,1.,4.,3.,8.])
#mat = np.array([[3.,4.,5.],[1.,2.,1.],[2.,1.,0.]])
#mat2 = np.array([[1.,4.,2.],[1.,1.,1.],[2.,1.,3.]])
#
#lengthV = np.linalg.norm(vec)
#transMat = mat.T #transpose, shortcut for np.transpose( )

""" more linear algebra """
#eigVals, eigVecs = np.linalg.eig(mat)
#newVec = np.dot(mat, vec.T)
#newMat = np.dot(mat, mat2.T)
#newMat2 = np.dot(mat, mat2)
#x, y, z = np.dot(mat, vec) #can save elements of any list or array as
                            #  individual elements

# These won't be needed if you are doing this at command line in interpreter
#print(lengthV)
#print(transMat)
#print(eigVals)
#print(eigVecs)
#print(eigVals[0])
#print(eigVecs[2])
#print(eigVecs[2,1])
#print(newVec)
#print(x, y, z)
#print(newMat)
#print(newMat2, '\n')
#print(vec5[0:2])       #note indices are zero-indexed and non-inclusive on the 
#print(vec5[1:4])       #  latter entry
#print(vec5[0:5])
#print(vec5[0:3])
#print(vec5[3:5])
#print()

""" Standard for loop use, demoed with above vectors. """
#for i in xrange(0,5):
#    print(i)
#    print(vec5[i])
#
#print()
#for i in xrange(0,5,2):   # multiple ways to generate lists of values, 
#    print(i)              #   for indexing, looping, etc. 
#    print(vec5[i])        #   also linspace, logspace
#
#print()
#print(np.arange(0,2,0.2))
#
#print()
#for elt in vec5:  #can iterate over a list, tuple, or array using this nice
#    print(elt)    #     syntax that doesn't indicate or care how many 
#                  #     elements are present.
#print()

""" Some basic function definitions with calls."""
def quad (x):
    out = x**2  # variable names in functions and loops are local scope
    return out

def mag (a,b,c):
    out = np.sqrt(a**2 + b**2 + c**2)
    return out

""" This one uses the splat operator *, which sends a specific list, tuple, or
array into a function as a set of individual elements. This particular function
is called using a vector splat, and the input arguments "unpack" the first two
elements and send the rest on as a black box to the "mag" function, where the
remaining three elements are finally unpacked. """
def calcSomething (a,b,*values):
    out = a/b * mag(*values)
    return out

#print(quad(3.))

#vecMag = mag (*vec)
#
#thing = calcSomething(*vec5)
#
#print(vecMag, '\n')
#print(thing)

""" The following creates an empty list dat, and then fills it with values read
from the file newdata.dat (provided). """
#dat = []
#f = open('newdata.dat','r+')
#print(f.readline(5))
#for line in f:
#    entry = [float(num) for num in line.split()]
#    dat.append(entry)
#
#dat = np.array(dat)
#print(dat)
#f.close()

""" Below, two ways to write new lines to dat file. First creates a new list
and appends formatted lines with entries from newdat array, then opens the dat
file in append mode and joins the newlines array to the existing file.

The latter opens the file for appending and adds a line directly. Note newdat
array is used in both examples. """
#newdat = np.array([[13.,56.],[14.,61.]])

#newlines = []
#for i in range(0,len(newdat)):
#    newlines.append ('{0:.6f} \t {1:.6f}'.format(
#                newdat[i,0], newdat[i,1]))
#
#with open('newdata.dat','a') as f:
#    f.write ('\n'.join (newlines))
#
#with open('newdata.dat','r+') as f:
#    dat = np.genfromtxt(f)
#
#print (dat)

#f = open('newdata.dat','a')
#f.write('{0:.6f}\t {1:.6f}'.format(newdat[0,0],newdat[0,1]))
#f.close
