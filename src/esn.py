import numpy as np
import types
import random
import string




###############################
# GRAPH NETWORK
# 
# Each Node is a population of continuous variables with a generic update function
# A Strategy design is adopted to flexibly assign different update functions to different nodes
# http://stackoverflow.com/questions/963965/how-is-this-strategy-pattern-written-in-python-the-sample-in-wikipedia
# http://codereview.stackexchange.com/questions/20718/strategy-design-pattern-with-various-duck-type-classes
#
# In order to access subset of connections, slicing is in place.
# A slice, assigned to an internal variable, is used as a view over the entire connections
# so it is possible to read or learn only a subset of connections.


def _reservoir( self, inputs ):
    self.state = (1 - self.tau) * self.state + self.tau * np.tanh( inputs )

def _load( self, inputs ):
    self.state = inputs




class Node( object ):

    def __init__( self, shape, tau, update_func, isSensory=False, isReadout=False ):
        self.isSensory = isSensory
        self.isReadout = isReadout
        self.tau = tau  # timestep execution
        self.update = types.MethodType( update_func, self, Node ) # strategy design
        self.shape = shape # N-dimesional shape
        self.state = np.zeros( shape ) # units' states


    def update( self ):
        pass # defined at init time




class Arc( object ):
    """
    Directed Arc

    Connections are organized by target units, to ease reading during execution.

    target : target node
    source : source node
    weight : base weight
    eps : learning speed
    """

    def __init__( self, target, source, weight, eps, isLearned=False ):
        self.isLearned = isLearned
        self.source = source
        self.target = target
        self.weight = weight
        self.eps = eps
        self.connections = np.array([])
        self.output = None
        # self.read = types.MethodType( read_func, self, Arc ) # strategy design
        # self.learn = types.MethodType( learn_func, self, Arc ) # strategy design


    def read( self, transpose=False ):
        """
        Reads the source state and multiply (dot) it by the connections defined to that source.
        """
        conns = self.connections 
        state = self.source.state
        # print conns.shape, state.shape
        # print self.target.n, self.source.n

        # is source is a single value
        if self.source.n == 1:
            self.output = conns * state
        # if the source is a numpy array:
        elif transpose:
            self.output = np.dot( conns.T, state )
        else:
            self.output = np.dot( conns, state )
 
        return self.output



    def learn( self, error ):
        """
        Learns the connections based on the source state and the error due to prediction mismatch.
        """
        conns = self.connections
        state = self.source.state
        conns += self.eps * state * error



    def initConnections( self, method=None, shape=None ):
        """
        Establishes the shape and initializes the numpy array of connections between source and target.
        The method passed is a numpy generator function.
        NOTE: the connection are always assumed to be ordered target-wise, so the first dimension of shape is the target.
        """

        self.connections = method( *shape ) # tuple content

        # apply weights
        self.connections *= self.weight

