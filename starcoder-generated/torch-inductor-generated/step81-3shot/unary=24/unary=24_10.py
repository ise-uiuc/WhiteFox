
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 9, (1,), stride=(1,), padding=(1,), dilation=(1,))
    def forward(self, x):
        negative_slope = -0.20666897
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.tensor([[-5.1749, -1.8530, -2.1135, -1.1023, -6.4949,  4.0892,  2.7016,  4.5393, -0.2191]])
