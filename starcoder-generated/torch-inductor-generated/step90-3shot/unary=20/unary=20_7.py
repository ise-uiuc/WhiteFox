
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.empty([1, 1024, 5, 5])
        kernel = torch.nn.init.kaiming_uniform_(kernel)
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=False)
        self.conv_t.weight = torch.nn.Parameter(kernel)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1024, 6, 6)
