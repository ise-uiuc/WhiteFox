
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x, x), dim=2).view(3, 6, 4)
        return x.sum()
# Inputs to the model
x = torch.randn(3, 2, 4, requires_grad=True)
