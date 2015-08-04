import numpy

class RandomController(object):

      def __init__(self,environment,model):
          self.environment = environment
          self.model = model
      
      def run(self,num_of_steps):
          angle = 0
          for i in xrange(0,num_of_steps):
              print "Step: ", i 
              angle = angle + (numpy.random.rand()-0.5) * 1.0
              self.model.run(self.environment.next_step([angle,10]))
      
            
            
            
