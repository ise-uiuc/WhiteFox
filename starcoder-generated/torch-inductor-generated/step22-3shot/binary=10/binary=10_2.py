
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.linear(16, 16)
        self.other = torch.randn(16, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(16, 16)
