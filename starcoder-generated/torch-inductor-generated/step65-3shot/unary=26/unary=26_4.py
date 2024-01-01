
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv_t = nn.ConvTranspose2d(2, 3, [3, 3], stride=1, padding=[0, 0], bias=False)

    def forward(self, x):
        x = self.conv_t(x)
        x = x > 0.5
        x = x * -0.125
        x = torch.where(x, x, x * -0.125)
        return x

# Inputs to the model
x1 = torch.randn(1,2,7,7)
