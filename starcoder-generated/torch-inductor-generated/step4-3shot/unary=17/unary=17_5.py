
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.deconv = nn.ConvTranspose2d(32, 12, kernel_size=2, stride=2, output_padding=0)
    def forward(self, x):
        e1 = self.deconv(x)
        e2 = F.relu(e1)
        return e2
# Inputs to the model
x = torch.randn(1, 32, 16, 16)
