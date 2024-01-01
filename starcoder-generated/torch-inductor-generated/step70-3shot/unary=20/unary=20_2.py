
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 73, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn = torch.nn.BatchNorm2d(num_features=11, eps=0.04571660119855066, momentum=0.1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.bn(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 6, 24)
