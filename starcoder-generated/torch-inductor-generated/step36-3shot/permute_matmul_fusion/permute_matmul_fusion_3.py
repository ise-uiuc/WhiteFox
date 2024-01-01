
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        temp = torch.bmm(v2, v1)
        v3 = temp.permute(0, 2, 1)
        temp1 = torch.bmm(v1, temp)
        v4 = temp1.permute(0, 2, 1)
        return v2, v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
