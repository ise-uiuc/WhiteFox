
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        out = self.linear(x1)
        out = out + 3
        out = torch.clamp_min(out, 0)
        out = torch.clamp_max(out, 6)
        out = out / 6
        return out

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3)
