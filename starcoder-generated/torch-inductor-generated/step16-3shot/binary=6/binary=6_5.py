
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 16)
        self.linear2 = torch.nn.Linear(16, 4)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = self.linear2(v1)
        v3 = v2 - 0.4
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
