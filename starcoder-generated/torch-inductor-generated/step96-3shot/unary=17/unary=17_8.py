
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(100, 200, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.ReLU(v1+x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 100, 224, 224)
x2 = torch.randn(1, 200, 56, 56)
