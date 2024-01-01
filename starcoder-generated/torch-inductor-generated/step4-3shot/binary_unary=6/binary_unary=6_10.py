
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(256, 10)
 
    def forward(self, x1):
        v1 = x1.view(x1.shape[0], -1)
        v2 = self.linear2(v1)
        v3 = v2 - other
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
