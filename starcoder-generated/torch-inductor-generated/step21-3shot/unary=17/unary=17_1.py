
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(3, 3, 1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(3, 1, 4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        t1 = torch.relu(v1)
        t2 = torch.sigmoid(t1)
        t3 = torch.tanh(t2)
        v2 = self.conv1(t3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
