import numpy
import pylab
import time

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
      def __init__(self,input_length):
          from minisom import MiniSom
          self.som = MiniSom(10, 10, input_length,sigma=0.3,learning_rate=0.1,normalize=True)
          

      def run(self,inp):
          self.som.trian_single_instance(inp.flatten())
