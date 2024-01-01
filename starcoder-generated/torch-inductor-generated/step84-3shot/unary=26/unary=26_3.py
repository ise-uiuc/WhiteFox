
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(506, 76, 2, stride=1, padding=0, bias=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = self.max_pool(x)
        v3 = torch.add(v1, v2)
        v4 = v3 > 0
        v5 = v3 * 2.1337
        v6 = torch.where(v4, v3, v5)
        return torch.nn.functional.relu(v6)
# Inputs to the model
x = torch.randn(1, 506, 151, 90)
