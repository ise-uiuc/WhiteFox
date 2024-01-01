
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 7)
        self.min_clamp = torch.nn.Hardtanh()
        self.max_clamp = torch.nn.Hardtanh(min_val=0, max_val=6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = self.min_clamp(v2)
        v4 = self.max_clamp(v3)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7)
