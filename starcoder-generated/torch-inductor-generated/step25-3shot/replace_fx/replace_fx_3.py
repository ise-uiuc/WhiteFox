
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.randn(1,2,3,2,1)
        x4 = torch.randn(3,2,1)
        x5 = torch.randn(2)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2,3,2,1)
