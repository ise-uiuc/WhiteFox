
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(512, 192, 3, stride=1, padding=0, output_padding=0)
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=-1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = self.maxpool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 103,106)
