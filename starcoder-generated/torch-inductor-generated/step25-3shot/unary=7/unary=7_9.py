
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, v1):
        v2 = self.linear(v1)
        v3 = v2 * torch.nn.functional.hardtanh(v2 + 3, 0, 6)
        return v3 / 6

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 3)
