import esn
import numpy as np

n = (10,10)
som_n = (3,)
out_n = (2,3)
tau = .1 # execution timestep for the cortical rate model
sigma = .001 # intra-reservoir weights
eps = .1 # learning rate

# Nodes: units, tau, method
SOM = esn.Node( som_n, 0, esn._load )
P = esn.Node( n, tau, esn._reservoir )
out = esn.Node( out_n, 0, esn._load )

# Arcs: target, source, weight, eps

# input from som
d_P  = esn.Arc( P, SOM, sigma, 0 )
d_P.initConnections( np.random.randn, P.shape+SOM.shape ) # type of init numpy func
print d_P.connections

# recurrent connections intra node
r_P  = esn.Arc( P, P, sigma, 0 )
r_P.initConnections( np.random.randn, P.shape+P.shape ) # type of init numpy func
print r_P.connections

# input from som
d_out  = esn.Arc( out, P, 0, eps )
print out.shape + P.shape
d_out.initConnections( np.random.randn, out.shape+P.shape ) # type of init numpy func
print d_out.connections

