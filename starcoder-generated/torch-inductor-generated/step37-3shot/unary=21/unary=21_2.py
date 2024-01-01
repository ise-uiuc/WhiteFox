
class ModelTanh(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        def forward(self, x):
                x = self.conv(x)
                x = torch.tanh(x)
                return x
# Inputs to the model
x = torch.randn(1, 3, 20, 20)
