
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 128)
        self.linear2 = torch.nn.Linear(256, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.randn(1, 32, 1, 1)
        v3 = self.linear2(v2) + torch.randn(1, 32, 1, 1)
        print(v3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 1, 1)
