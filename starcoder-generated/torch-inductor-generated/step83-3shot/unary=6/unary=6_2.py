
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0, num_args=2)
        t4 = torch.clamp_max(t3, 6, num_args=2)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
