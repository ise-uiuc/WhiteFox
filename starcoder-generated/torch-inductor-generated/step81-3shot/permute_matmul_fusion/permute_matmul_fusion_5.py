
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        permute = x1.permute(0, 2, 1)
        bmm = torch.bmm(permute, x2)
        return bmm
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
