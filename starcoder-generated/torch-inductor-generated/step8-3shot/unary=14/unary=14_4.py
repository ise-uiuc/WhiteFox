
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(11, 11), stride=1, padding=(7, 7))
        self.convtranspose1 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=(1, 1), output_padding=0)
        self.sigmoid = nn.Sigmoid()
        self.mul1 = F.mul
        
    def forward(self, x):
        v1 = self.convtranspose1(self.conv1(x))
        v2 = self.sigmoid(v1)
        v3 = self.mul1(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
