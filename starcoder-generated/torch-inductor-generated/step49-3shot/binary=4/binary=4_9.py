
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(640, 128)
        self.linear2 = torch.nn.Linear(128, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = self.linear2(v2)
        v4 = v3 + x1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 640)
x2 = torch.randn(4, 64)
