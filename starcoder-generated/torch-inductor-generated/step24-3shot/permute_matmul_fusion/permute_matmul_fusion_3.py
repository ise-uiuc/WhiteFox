
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, x)
        x = x.permute(0, 2, 1).contiguous()
        out1 = x
        out2 = x
        out3 = x
        return out1, out2, out3
# Inputs to the model
x = torch.randn(2, 2, 2, requires_grad=True)
