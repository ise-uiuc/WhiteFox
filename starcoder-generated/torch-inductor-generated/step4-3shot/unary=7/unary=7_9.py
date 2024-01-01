
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 16)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(self.linear(x1) + 3, 0, 6)
        s2 = self.sigmoid(v2)
        return s2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
