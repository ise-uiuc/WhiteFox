
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(2, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return torch.squeeze(v5, dim=0)
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
