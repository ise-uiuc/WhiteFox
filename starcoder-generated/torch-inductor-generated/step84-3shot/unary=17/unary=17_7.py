
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, 3)
        self.conv2 = torch.nn.ConvTranspose2d(3, 3, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = torch.sigmoid(v3)  
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
