
class Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, num_classes, kernel_size=(5,5), bias=False)
    def forward(self, x1):
        x1 = self.conv_t(x1)
        x1 = torch.sigmoid(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 650, 255)
