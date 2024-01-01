
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=1, padding=1, groups=1, bias=False)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x, x], dim=1)
        return x
# Inputs to the model. 16 random values from a normal distribution with std==1. The normal distribution is used to avoid zero values
x = torch.randn(16, 3, 3, 3)
