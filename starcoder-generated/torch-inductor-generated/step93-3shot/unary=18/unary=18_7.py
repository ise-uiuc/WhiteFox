
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(15, 4, (5, 1, 1), stride=(3, 1, 2), padding=(4, 0, 1))
        self.conv2 = torch.nn.Conv3d(8, 6, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 15, 33, 35, 39)
