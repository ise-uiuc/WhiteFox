
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=15, kernel_size=(3, 5), stride=(1, 2), padding=(2, 0))
        self.conv1_2 = torch.nn.ConvTranspose2d(15, 1, 1)
    def forward(self, x1):
        v1 = self.conv1_1(x1)
        v2 = self.conv1_2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 15, 10, dtype=torch.float32)
