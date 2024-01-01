
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(479, 11, 3, stride=2)
        self.negative_slope = negative_slope
    def forward(self, x1):
        t1 = self.conv_t(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.hardtanh(torch.nn.functional.relu(t4))
negative_slope = 0.74
# Inputs to the model
x1 = torch.randn(6, 479, 38, 20)
