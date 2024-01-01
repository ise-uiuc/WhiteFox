
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(640, 640)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - 0.7071067811865476
        v3 = v2 - x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 640, 32, 32)
x2 = torch.randn(1, 640, 1, 1, requires_grad=True)
