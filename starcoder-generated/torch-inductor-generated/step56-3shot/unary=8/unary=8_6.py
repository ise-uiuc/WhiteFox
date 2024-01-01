
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1024, 1, 2, stride=(2,1), padding=1)
        self.bn = torch.nn.BatchNorm1d(4096)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.bn(v1.flatten(1))
        v3 = v2.view(-1, 6, 16, 2, 2)
        v4 = self.relu(v3)
        v5 = v4.flatten(2,3).flatten(0,1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1024, 21, 24, 24)
