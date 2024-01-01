
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.bmm(x, x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
