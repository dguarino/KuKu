import numpy

class RandomController(object):

      def __init__(self,environment,model,store_history=False):
          self.environment = environment
          self.model = model
          self.store_history = store_history
          self.history = []
      
      def run(self,num_of_steps):
          angle = 0
          for i in xrange(0,num_of_steps):
              print "Step: ", i 
              angle = angle + (numpy.random.rand()-0.5) * 1.0
              env_out = self.environment.next_step([angle,10])
              if self.store_history:
                  self.history.append(env_out)
              self.model.run(numpy.append(env_out.flatten(),[angle,10]))
      
            
            
            
