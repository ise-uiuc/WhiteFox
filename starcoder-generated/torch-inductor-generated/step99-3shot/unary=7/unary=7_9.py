
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp(torch.nn.functional.linear(x, v1), min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(64, 512)
