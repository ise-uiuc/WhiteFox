
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1d1 = torch.nn.ConvTranspose1d(512, 64, kernel_size=1, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
        self.convtranspose1d2 = torch.nn.ConvTranspose1d(64, 256, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
        self.linear = torch.nn.Linear(256, 17)
    def forward(self, x1, x2):
        x1 = self.convtranspose1d1(x1)
        x1 = self.convtranspose1d2(x1)
        x1 = self.linear(x1)
        x1 = torch.relu(x1)
        x1 = x1.add_(x2)
        x1 = torch.sigmoid(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 512, 5)
x2 = torch.randn(1, 5, 40)
