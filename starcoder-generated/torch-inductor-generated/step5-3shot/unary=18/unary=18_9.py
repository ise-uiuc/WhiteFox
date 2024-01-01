
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
    def forward(self, x1):
        v1 = self.relu(self.upsample(x1))
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
