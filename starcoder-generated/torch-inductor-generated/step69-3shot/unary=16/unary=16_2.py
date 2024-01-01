
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(32 * 32 * 1, 100))
        self.layers.add_module('relu1', torch.nn.ReLU(inplace=True))
        self.layers.add_module('fc2', torch.nn.Linear(100, 10))

    def forward(self, x1):
        v1 = self.layers(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(256, 32 * 32 * 1)
