
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(4, 8, 1, stride=1, padding=2)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = self.relu(t1+3)
        t3 = F.hardtanh(t2/5, min_val=0.0, max_val=6.0)
        return t3
# Inputs to the model
x1 = torch.randn(1, 4, 288, 288)
