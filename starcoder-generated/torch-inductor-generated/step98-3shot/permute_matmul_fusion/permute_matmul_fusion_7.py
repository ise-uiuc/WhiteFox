
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1
        v1 = x2.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        return torch.matmul(v1, v2) # change to torch.bmm
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
