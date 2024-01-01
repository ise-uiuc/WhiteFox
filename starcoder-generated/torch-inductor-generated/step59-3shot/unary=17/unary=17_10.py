
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(3, 5, (2, 2))
        self.conv2 = nn.ConvTranspose2d(5, 1, (3, 3))
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = F.sigmoid(self.conv2(x1))
        return x1, x2
# Inputs to the model
x  =  torch.randn(1, 3, 128, 128)
