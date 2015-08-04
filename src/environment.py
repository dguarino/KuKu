import numpy
from PIL import Image



class Environment(object):

    def next_step(self,motor_command):
        """
        This takes in a vector corresponding to the motor commands.
        This should return 2D ndarray containing the pixels that the entity sees 
        in the next step given the motor command.
        """
        raise NotImplementedError;

class StaticImageEnvironment(Environment):
    """
    This environment expects motor command of two values, the angle of the movement, and the force of the 
    movement (in pixels per step).
    """
    
    def __init__(self,image_path,viewport_size):
        """
            image_path : path to the image that will constitute the environment
            viewport_size : size of the view port in pixels, it has to be divisible by two
        """
        self.im = numpy.asarray(Image.open(image_path))
        self.sizex, self.sizey = numpy.shape(self.im)
        self.x = self.sizex/2.0
        self.y = self.sizey/2.0
        self.viewport_size = viewport_size
        assert viewport_size % 2 == 0
        
    def next_step(self,motor_command):        
        assert len(motor_command) == 2
        print self.x
        self.x += numpy.cos(motor_command[0]) * motor_command[1]
        self.y += numpy.sin(motor_command[0]) * motor_command[1]
        
        # make sure we do not run out of image
        self.x = max(self.viewport_size/2,self.x)
        self.y = max(self.viewport_size/2,self.y)
        self.x = min(self.sizex-self.viewport_size/2,self.x)
        self.y = min(self.sizey-self.viewport_size/2,self.y)
        
        return self.im[self.x-self.viewport_size/2:self.x+self.viewport_size/2,self.y-self.viewport_size/2:self.y+self.viewport_size/2]

