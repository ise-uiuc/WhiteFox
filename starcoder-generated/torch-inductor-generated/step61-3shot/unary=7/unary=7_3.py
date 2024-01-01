
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16 * 5 * 5, 120)
 
    def forward(self, x1):
        c1 = x1.reshape((-1, 16 * 5 * 5)).to(device=device)
        v1 = self.linear(c1)
        v2 = torch.clamp(torch.add(v1, 3), min=0, max=6) * 6
        return v2

# Initializing the model
x = torch.randn(1, 16 * 5 * 5)
m = Model()

# Inputs to the model
