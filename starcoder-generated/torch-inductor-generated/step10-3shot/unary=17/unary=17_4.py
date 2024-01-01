
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)
        self.conv_1.padding_mode ='replicate'
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
