
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(1, 0, 2).unsqueeze_(2)
        v2 = torch.bmm(x1.permute(1, 2, 0), v1)
        v3 = v2.squeeze_()
        v4 = v3.transpose(1, 2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
