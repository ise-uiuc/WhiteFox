
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.dropout_layer = torch.nn.Dropout()
    def forward(self, x1):
        x2 = self.conv2d(x1)
        x3 = self.dropout_layer(x2)
        x4 = torch.rand_like
        return x4
# Inputs to the model
x1 = torch.randn(2, 128, 1, 1)
