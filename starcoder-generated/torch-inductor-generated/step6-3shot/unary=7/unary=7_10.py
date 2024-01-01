
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 512)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(v1 + 3, max=6)
        t3 = v2 / 6
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
