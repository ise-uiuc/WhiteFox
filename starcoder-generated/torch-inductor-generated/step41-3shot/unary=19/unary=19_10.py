
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(6)
        self.w = torch.nn.Parameter(self.w)
 
    def forward(self, x2):
        v1 = torch.matmul(x2, self.w)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
