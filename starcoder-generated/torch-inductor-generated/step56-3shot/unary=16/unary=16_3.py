
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 2048)
 
    def forward(self, x1):
        v0 = x1.view(x1.size(0), -1)
        v1 = self.linear(v0)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048)
