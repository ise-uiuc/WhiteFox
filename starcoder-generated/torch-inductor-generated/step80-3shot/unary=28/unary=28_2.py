
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(6, 12)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu6(v1, inplace=True)
        return torch.square(v2)

# Initializing the model
min = 0
max = 1
m = Model(min, max)

# Inputs to the model
x1 = torch.randn(1, 6)
