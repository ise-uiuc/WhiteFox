
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d1 = torch.nn.Conv1d(56, 75, 3, stride=2, padding=1)
        self.conv1d2 = torch.nn.Conv1d(56, 73, 1, stride=1, padding=1)
    def forward(self, t1, x3, t0):
        v1 = self.conv1d1(t1)
        v2 = self.conv1d2(v1)
        if t0:
            x3 = v2
        else:
            t2 = t1 + x3
        t3 = t2.view(-1)
        return t1
# Inputs to the model
t1 = torch.randn(25, 56, 26)
x3 = torch.randn(1, 73, 26)
t0 = 0
