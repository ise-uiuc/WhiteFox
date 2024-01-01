
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.tanh_1 = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = torch.tanh_1(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
