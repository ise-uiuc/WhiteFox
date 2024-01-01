
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, kernel_size=(5, 5), stride=(2, 2))
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, kernel_size=(5, 5), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_t(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 123, 51)
