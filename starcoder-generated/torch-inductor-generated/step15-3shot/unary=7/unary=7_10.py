
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        #TODO: Fix me!
        v2 = v1 * torch.clamp(v1 + 3, min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing and running the model
m = Model()
x1 = torch.randn(1, 10)
