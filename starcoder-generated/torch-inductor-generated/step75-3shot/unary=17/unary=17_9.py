
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose1 = torch.nn.ConvTranspose2d(in_channels=5, out_channels=4, kernel_size=5, stride=2, padding=1)
        self.convTranspose2 = torch.nn.ConvTranspose2d(in_channels=5, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.convTranspose3 = torch.nn.ConvTranspose2d(in_channels=4, out_channels=6, kernel_size=2, stride=2, padding=0)
        self.convTranspose4 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.convTranspose1(x1)
        v2 = torch.relu(v1)
        v3 = self.convTranspose2(x1)
        v4 = torch.relu(v3)
        v5 = self.convTranspose3(v4)
        v6 = torch.relu(v5)
        v7 = self.convTranspose4(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 5, 28, 28)
