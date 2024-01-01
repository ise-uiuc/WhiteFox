
class Model(torch.nn.Module):
    def __init__(self, min_value=0.005, max_value=0.6):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2):
        t1 = self.convt(x1)
        t2 = torch.clamp(t1, self.min_value, self.max_value)
        t3 = self.convt(t2)
        t4 = torch.clamp(t3, self.min_value, self.max_value)
        t5 = t4 + x2
        t6 = t5 * x2
        return t6
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
x2 = torch.randn(1, 1, 3, 3)
