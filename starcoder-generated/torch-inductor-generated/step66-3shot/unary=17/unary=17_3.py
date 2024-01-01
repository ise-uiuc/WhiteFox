
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspo = torch.nn.ConvTranspose2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.convTranspo1 = torch.nn.ConvTranspose2d(32, 2, kernel_size=2, stride=1, padding=0)
        self.convTranspo2 = torch.nn.ConvTranspose2d(2, 32, kernel_size=3, stride=2, padding=1)
        self.convTranspo3 = torch.nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.convTranspo(x1)
        v2 = torch.relu(v1)
        v3 = self.convTranspo1(v2)
        v4 = torch.relu(v3)
        v5 = self.convTranspo2(v4)
        v6 = torch.relu(v5)
        v7 = self.convTranspo3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
