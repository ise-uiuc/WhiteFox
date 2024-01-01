
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
    def forward(self, x):
        x = self.model(x)
        return x
# Inputs to the model
x = torch.randn(128, 32, 10, 10)
