
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Parameter(torch.randn(300, 400))
 
    def forward(self, x1):
        v2 = torch.matmul(x1, self.v1)
        return v2 + torch.randn(1, 300)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 400)
