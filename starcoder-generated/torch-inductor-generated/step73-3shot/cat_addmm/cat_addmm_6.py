
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Conv2d(in_channels=7, out_channels=10, kernel_size=(5, 3))
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(64, 7, 37, 50)
