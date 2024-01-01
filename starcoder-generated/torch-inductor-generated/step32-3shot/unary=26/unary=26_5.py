
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Conv1d(35, 35, 2, stride=1, padding=1)
        self.neg = torch.nn.Neg()
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = self.neg(y1)
        y3 = y1 > 0
        y4 = y1 * 3.654
        y5 = torch.where(y3, y1, y4)
        return y5
# Inputs to the model
x = torch.randn(64, 35, 34, device='cuda')
