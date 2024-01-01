
class Model(torch.nn.Module):
    def __init__(self):
        super(Module):
            self.layer1 = torch.nn.Linear(10, 10)
            self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.layer1(x)
        v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
