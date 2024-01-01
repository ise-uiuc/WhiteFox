
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
 
    def forward(self, l1):
        v1 = self.linear(l1)
        v2 = v1 * torch.clamp(min=0, max=6, v1 + 3) 
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 128)
