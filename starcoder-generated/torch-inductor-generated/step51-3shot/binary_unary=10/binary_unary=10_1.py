
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2520,8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.empty_like(v1).normal_()
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 2520)
