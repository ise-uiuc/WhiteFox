
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(3, 8, kernel_size=3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4, 4)
