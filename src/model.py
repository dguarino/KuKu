import numpy
import pylab
import time

pylab.figure()
pylab.ion()
pylab.draw()


class Model(object):

    def run(self,input):
        pylab.cla()
        pylab.imshow(input,cmap='gray',interpolation="none")
        pylab.draw()
        time.sleep(0.05) # delays for 5 seconds
        
        
