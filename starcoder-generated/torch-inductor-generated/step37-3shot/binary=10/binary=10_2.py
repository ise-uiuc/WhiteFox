
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
    
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.add(other=1e-8)
        return v2
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
