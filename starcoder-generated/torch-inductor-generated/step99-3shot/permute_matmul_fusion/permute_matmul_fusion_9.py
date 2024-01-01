
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.bmm(x, x.permute(0, 2, 1))
# Inputs to the model
x = torch.randn(1, 2, 2)
