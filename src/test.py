import esn
import numpy as np


# parameters
T = 400
res_n = (10,)
som_n = (3,)
out_n = (3,)
tau = .1 # execution timestep for the cortical rate model
sigma = .001 # intra-reservoir weights
eps = .1 # learning rate


# Build the network
# Nodes: units, tau, method
SOM = esn.Node( som_n, 0, esn._load )
P = esn.Node( res_n, tau, esn._reservoir )
out = esn.Node( out_n, 0, esn._load )

# Arcs: target, source, weight, eps

# input from som
d_P  = esn.Arc( P, SOM, sigma, 0 )
d_P.initConnections( np.random.randn, P.shape+SOM.shape ) # type of init numpy func
#print d_P.connections

# recurrent connections intra node
r_P  = esn.Arc( P, P, sigma, 0 )
r_P.initConnections( np.random.randn, P.shape+P.shape ) # type of init numpy func
#print r_P.connections

# input from som
d_out  = esn.Arc( out, P, 0, eps )
# print out.shape + P.shape
d_out.initConnections( np.random.randn, out.shape+P.shape ) # type of init numpy func
#print d_out.connections


###################
# Main loop
print "SOM.state:",SOM.state
print "P.state:",P.state
print "out.state:",out.state
# problem: after the first read, the state is different due to matrix multiplication

for t in range(T):
    print "\n\ntime:", t

    # INPUT
    # retrieve input for this timestep
    # SOM.update( np.array([t]) ) # 1
    SOM.update( np.array([t,t+1,t+2]) ) # 3
    # print "SOM.state:",SOM.state

    print "d_P.read():", d_P.read()
    P.update( d_P.read() )
    # print "P.state:",P.state

    # print "d_out.read():", d_out.read()
    out.update( d_out.read() )
    # print "out.state:",out.state

    # LEARNING
    print "error:",SOM.state - out.state
    d_out.learn( SOM.state - out.state )
    # print "d_out.connections:"
    # print d_out.connections

    # RECORDING
    #tot_z = d_aA.output + d_bB.output + d_cC.output
    #tot_e = u - tot_z
