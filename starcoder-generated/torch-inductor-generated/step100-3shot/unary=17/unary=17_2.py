
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_2 = torch.nn.ConvTranspose2d(1, 16, 2, stride=1, padding=0)
        self.conv2d = torch.nn.ConvTranspose2d(16, 4, 1, stride=2)

    def forward(self, input1):
        v1 = self.conv2d_2(input1)
        v2 = torch.relu(v1)
        v3 = self.conv2d(v2)
        return torch.relu(v3)
# Inputs to the model
input1 = torch.randn(1, 1, 3, 3)
