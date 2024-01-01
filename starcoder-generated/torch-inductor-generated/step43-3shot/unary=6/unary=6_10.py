
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(1, 3), stride=(1,1), padding=0, dilation=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = t1.sum((2, 3))
        t3 = t2 + 3
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t1 * t5
        t7 = t6 /6
        return t7
# Inputs to the model
x1 = torch.randn(2, 3, 120, 120)
