
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v3 = v1 + x2
        v2 = torch.nn.ReLU()
        return v2(v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 10)
