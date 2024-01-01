
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x.permute(1, 0, 2)
        v2 = torch.bmm(v1.permute(2, 1, 0), x)
        return v2
# Inputs to the model
x = torch.randn(2, 2, 2)
