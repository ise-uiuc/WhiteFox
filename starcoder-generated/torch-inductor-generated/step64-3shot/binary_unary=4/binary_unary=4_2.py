
class Model(torch.nn.Module):
    def __init__(self):
        self.linear = torch.nn.Linear(12, 8)
 
    def forward(self, x):
        v = self.linear(x)
        v = v + x
        return torch.relu(v)
 
# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(12, 8)
