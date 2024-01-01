
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(2, 1, 2), padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 16, 16, 16)
