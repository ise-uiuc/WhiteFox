
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 2, 2, stride=2, bias=False)
        self.relu = torch.nn.ReLU6()
    def forward(self, x):
        t3 = self.conv_t(x)
        t35 = t3 > 0
        t38 = t3 * -0.2128
        t39 = torch.where(t35, t3, t38)
        t40 = t39.neg()
        return self.relu(t40)
# Inputs to the model
x = torch.randn(1, 5, 3, 3)
