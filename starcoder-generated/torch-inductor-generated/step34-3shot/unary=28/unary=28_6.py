
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        y1 = self.linear1(x1)
        v1 = torch.clamp_min(y1, min=)
        v2 = torch.clamp_max(v1, max=)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
