
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 3
    def forward(self, x):
        y = torch.cat((x + 1.0, x.transpose(0, 1)), dim=self.dim)
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
