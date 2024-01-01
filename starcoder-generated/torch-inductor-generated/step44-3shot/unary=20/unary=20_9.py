
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = Conv2dT(32, 3, 2, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv_t(x)
        x = self.sigmoid(x)
        return x
# Input to the model
x = torch.randn(1, 32, 32, 32)
