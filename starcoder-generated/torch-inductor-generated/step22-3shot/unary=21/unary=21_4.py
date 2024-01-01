
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(3, 32, kernel_size=(1, 3), stride=(1, 2))
        self.n25 = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(2, 2))
    def forward(self, x):
        v0 = torch.tanh(self.conv_0(x))
        v1 = self.n25(v0)
        v2 = torch.tanh(self.conv_1(v1))
        return v2
# Inputs to the model
x0 = torch.randn(20, 3, 128, 67)
