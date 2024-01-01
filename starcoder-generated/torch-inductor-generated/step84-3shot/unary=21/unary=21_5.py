
# Use the Sequential with Conv2d, and hard code the activation functions.
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(7, 10, kernel_size=(5, 5)),
            torch.nn.Tanh()
        )
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(64, 7, 128, 128)
