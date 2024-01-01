
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
