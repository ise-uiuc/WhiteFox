
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 10)
 
    def forward(self, x2, o2):
        v1 = self.linear(x1)
        v2 = v1 + o2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()
m.eval():

# Inputs to the model
x2 = torch.randn(1, 8, 1, 32)
o2 = torch.randn(1, 10)
