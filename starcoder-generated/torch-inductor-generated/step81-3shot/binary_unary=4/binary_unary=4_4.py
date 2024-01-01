
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
        self.other = other
 
    def forward(self, x2):
        v3 = self.linear(x2)
        v7 = v3 + self.other
        v8 = torch.relu(v7)
        return v8
     
# Initializing the model
m = Model(torch.randn(64))

# Inputs to the model
x2 = torch.randn(1, 64)
__output = m(x2)

