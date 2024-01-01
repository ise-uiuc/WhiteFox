
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mean(x1, dim=[2, 3], keepdim=True)
        v2 = x2 + v1
        return v2
# Inputs to the model
x1 = torch.randn(100, 1024, 20)
x2 = torch.randn(100, 1024, 20)
