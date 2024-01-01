
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 2, 2, stride=2)
        self.relu = torch.nn.ReLU6()
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.relu(t4)
        return (t5 - 1.0) * -5
negative_slope = 0.17
# Inputs to the model
x = torch.randn(6, 2, 3, 3)
