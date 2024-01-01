
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layer = nn.ConvTranspose2d(4, 3, kernel_size=3, stride=[1,1], output_padding=[0,0])
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x1 = layer(x)
        x2 = self.sig(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 4, 12, 12, requires_grad=True)
