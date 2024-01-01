
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1_permute = x1.permute(2, 0, 1)
        x1_permute = x1_permute.permute(2, 0, 1).mul(x1_permute)
        x1_permute = x1_permute.permute(2, 0, 1)
        x2_permute = x2.permute(0, 2, 1)
        x2_permute.div(x2_permute)
        x2_permute = x2_permute.permute(2, 0, 1)
        x2_permute = x2_permute.permute(2, 0, 1).div(x2_permute)
        x2_permute = x2_permute.permute(0, 2, 1)
        res = torch.bmm(x1_permute, x2_permute)
        return res
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 2, 2)
x3 = torch.randn(1, 2, 1)
x4 = torch.randn(1, 2, 2)
