from environment import StaticImageEnvironment
from controller import RandomController
from model import *
import numpy

env = StaticImageEnvironment("../assets/image.jpg",10)
model = KuKuModel(2,100,500)

rc = RandomController(env,model,store_history=True)
rc.run(10000)

wm=rc.model.som.win_map_closest([a.ravel() for a in rc.history])

fig = numpy.zeros((100,100))

for (i,j) in wm.keys():
    fig[i*10:(i+1)*10,j*10:(j+1)*10] = numpy.reshape(wm[(i,j)],(10,10))


pylab.imshow(fig,cmap='gray')
pylab.show()
