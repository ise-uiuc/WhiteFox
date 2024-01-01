
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 2, 2, stride=2, bias=False)
        self.relu = torch.nn.ReLU6()
    def forward(self, x2):
        o1 = self.conv_t(x2)
        o2 = o1 * 0.108
        o3 = torch.where(o2 > 0, o2, o1)
        o4 = self.relu(o3)
        return torch.nn.functional.adaptive_avg_pool2d(o4, (1, 1))
# Inputs to the model
x2 = torch.randn(19, 1, 5, 15)
