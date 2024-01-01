
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(36, 10)
 
    def forward(self, x1):
        v2 = x1.view(-1, 36)
        v3 = self.linear(v2)
        v4 = v3 + x1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
