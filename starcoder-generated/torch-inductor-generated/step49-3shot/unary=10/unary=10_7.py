
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(300, 400, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 300)

