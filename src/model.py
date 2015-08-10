import numpy
import pylab
import time
import esn

class Model(object):

    def run(self,inp):
        raise NotImplementedError;




class SomModel(Model):
      def __init__(self,input_length):
          from minisom import MiniSom
          self.som = MiniSom(10, 10, input_length,sigma=0.3,learning_rate=0.1,normalize=True)

      def run(self,inp):
          self.som.trian_single_instance(inp.flatten())
 
class KuKuModel(Model):
    
      def __init__(self,proprioception_input_length,sensory_input_length,reservoir_size):
            # Build the Reservoir
            tau = .1 # execution timestep for the cortical rate model
            sigma = .001 # intra-reservoir weights
            eps = .1 # learning rate
            som_size = 10*10
            self.sensory_input_length = sensory_input_length
            self.proprioception_input_length = proprioception_input_length
            
            full_reservoir_input_length = proprioception_input_length+som_size
            # Nodes: units, tau, method
            self.reservoir_input = esn.Node((full_reservoir_input_length,), 0, esn._load )
            self.reservoir = esn.Node((reservoir_size,), tau, esn._reservoir )
            self.reservoir_output = esn.Node((som_size,), 0, esn._load )
            
            # Arcs: target, source, weight, eps
            # input from som
            self.d_P  = esn.Arc( self.reservoir, self.reservoir_input, sigma, 0 )
            self.d_P.initConnections( numpy.random.randn, self.reservoir.shape+self.reservoir_input.shape ) # type of init numpy func
            #print d_P.connections

            # recurrent connections intra node
            self.r_P  = esn.Arc( self.reservoir, self.reservoir, sigma, 0 )
            self.r_P.initConnections( numpy.random.randn, self.reservoir.shape+self.reservoir.shape ) # type of init numpy func
            #print r_P.connections

            # input from som
            self.d_out  = esn.Arc( self.reservoir_output, self.reservoir, 0, eps )
            self.d_out.initConnections( numpy.random.randn, self.reservoir_output.shape+self.reservoir.shape ) # type of init numpy func
            #print d_out.connections
                      
            from minisom import MiniSom
            self.som = MiniSom(10, 10, sensory_input_length,sigma=0.3,learning_rate=0.1,normalize=True)
            
            self.previous_som_activation = numpy.zeros((10,10))
          

      def run(self,inp):
          self.som.train_single_instance(inp[:self.sensory_input_length])
          
          self.reservoir_input.update(numpy.append(self.previous_som_activation.flatten().copy(),inp[-self.proprioception_input_length:])) # 3
          self.reservoir.update(self.d_P.read())
          self.reservoir_output.update(self.d_out.read())
          print "error:",self.som.activation_map.flatten()  - self.reservoir_output.state
          self.d_out.learn(self.som.activation_map.flatten()  - self.reservoir_output.state )
          
          self.previous_som_activation = self.som.activation_map.flatten().copy()
          
          
          
          
