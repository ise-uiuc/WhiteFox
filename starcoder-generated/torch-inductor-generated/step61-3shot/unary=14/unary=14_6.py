
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv8 = torch.nn.Conv2d(in_channels=299, out_channels=5, kernel_size=(5, 1), stride=(1, 1),
                                    padding=(2, 0))
    def forward(self, x1):
        v1 = self.conv8(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 299, 102, 3)
