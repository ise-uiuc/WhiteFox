
class ModelWithConstantReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        return y.view(7, 2, -1).tanh()
x = torch.randn(3, 2, 4)
