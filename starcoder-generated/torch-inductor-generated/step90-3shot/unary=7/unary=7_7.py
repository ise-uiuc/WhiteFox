
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.min(v1, 6), 0, 6) + 3
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()
x1 = torch.randn(64, 1)
