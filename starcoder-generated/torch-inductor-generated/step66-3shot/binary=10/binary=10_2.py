, with the attribute self.other initialized to the specified tensor
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.randn(1, 16, 8, 8)
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.ones(16, 3), bias=None)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
