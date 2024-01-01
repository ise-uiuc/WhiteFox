
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=0, padding1=0):
        v1 = self.conv(x1)
        if padding1 == 0:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(3, 8, 128)
