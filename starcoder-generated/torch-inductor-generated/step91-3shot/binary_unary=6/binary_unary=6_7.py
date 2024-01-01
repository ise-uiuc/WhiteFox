
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear= torch.nn.Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5420211607450407
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 8)
