
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=2, padding=0, transposed=True)
        self.conv2d = torch.nn.ConvTranspose2d(5, 3, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.relu(v1)
        v3 = v2.transpose(2, 1)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 480, 100)
