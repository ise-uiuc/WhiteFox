
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.add(torch.nn.Conv2d(3, 4, 5, stride=1, padding=2), 28)
    def forward(self, x1):
        q1 = self.conv(x1)
        q2 = q1 + 3
        q3 = torch.clamp(q2, 0, 6)
        q4 = q1 * q3
        q5 = q4 / 6
        return q5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
