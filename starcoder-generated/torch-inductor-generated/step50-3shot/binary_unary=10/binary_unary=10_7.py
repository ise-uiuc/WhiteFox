
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.randn(8)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(67108864)
