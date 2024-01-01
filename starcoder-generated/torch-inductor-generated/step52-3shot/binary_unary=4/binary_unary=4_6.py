
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is not None:
            v1 += other
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
other = torch.randn(2, 4)
