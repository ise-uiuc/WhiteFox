
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 2, bias=False)
 
    def forward(self, x1):
        v1 = self.net(x1)
        v2 = v1 - 3.14
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
