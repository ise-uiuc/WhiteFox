
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = torch.addmm(x1, self.linear.weight, self.linear.bias)
        v2 = v1.unsqueeze(dim=0)
        v3 = torch.cat([v2], dim=0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 3)
