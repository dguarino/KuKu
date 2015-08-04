from environment import StaticImageEnvironment
from controller import RandomController
from model import Model 

env = StaticImageEnvironment("../assets/image.jpg",50)
model = Model()

RandomController(env,model).run(100)
