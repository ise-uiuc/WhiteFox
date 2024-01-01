
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(256, 32, (1,), stride=[1], bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1.sigmoid()
        return v1 * v2
# Inputs to the model
x2 = torch.randn(1, 256, 64)
