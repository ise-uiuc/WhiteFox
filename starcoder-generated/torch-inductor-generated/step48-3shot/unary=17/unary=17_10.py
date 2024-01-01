
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose3d(1, 10, kernel_size=(2, 4, 6), stride=(2, 2, 2))
        self.conv2 = torch.nn.ConvTranspose3d(10, 20, kernel_size=(2, 3, 5), stride=(1, 3, 1))
        self.conv3 = torch.nn.ConvTranspose3d(20, 1, kernel_size=(2, 2, 4), stride=(1, 3, 4))
    def forward(self, x0):
        v0 = self.conv1(x0)
        v1 = torch.relu(v0)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(torch.unsqueeze(v3, dim=0))
        return torch.tanh(torch.squeeze(v4, dim=0))
# Inputs to the model
x0 = torch.randn(1, 1, 16, 32, 32)
