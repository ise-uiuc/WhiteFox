
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        self.other = torch.rand(1, 16)
 
    def forward(self, x1):
        v1 = torch.relu(self.linear(x1) + self.other)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
