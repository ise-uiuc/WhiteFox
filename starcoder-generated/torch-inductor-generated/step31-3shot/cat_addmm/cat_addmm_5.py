
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.addmm
    def forward(self, x):
        x = self.layers(x, x, x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
