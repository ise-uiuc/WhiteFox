
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.bmm(v1, v1).permute(0, 2, 1)
        return v2
# Inputs to the model
x = torch.randn(2, 2, 2, requires_grad=True)
