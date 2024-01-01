
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(1, 10, 4, padding=1)
        self.conv  = torch.nn.Conv2d(10, 1, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(4, 7, kernel_size=(2,3), padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(7, 8, kernel_size=(4,3), padding=1)
    def forward(self, x1):
        v2 = torch.relu(self.convT(x1))
        v3 = torch.sigmoid(self.conv(v2))
        v1 = torch.relu(self.conv1(x1))
        v2 = self.conv2(v1)
        v2 = torch.sigmoid(v2)
        result = v2 * v3
        return result
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
