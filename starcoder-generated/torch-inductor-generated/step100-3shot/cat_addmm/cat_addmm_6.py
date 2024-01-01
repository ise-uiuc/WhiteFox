
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn((2, 2)))
    def forward(self, x):
        y = x + self.param
        return y
# Inputs to the model
x = torch.randn(2, 2)
