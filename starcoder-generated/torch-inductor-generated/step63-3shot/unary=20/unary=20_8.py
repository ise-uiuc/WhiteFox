
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_first_layer = torch.nn.MaxPool2d((12, 19), padding=(11, 10))
        self.conv_t = torch.nn.ConvTranspose2d(23, 64, kernel_size=8, stride=4)
    def forward(self, x1):
        v1 = self.max_pool_first_layer(x1)
        v2 = self.conv_t(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 23, 37, 42)
