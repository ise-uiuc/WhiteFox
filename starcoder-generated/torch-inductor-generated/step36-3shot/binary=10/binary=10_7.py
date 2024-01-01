
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 2)
 
    def forward(self, x):
        fc1_x = self.fc1(x)
        fc2_x = self.fc2(fc1_x)
        return fc2_x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn((10,3))
