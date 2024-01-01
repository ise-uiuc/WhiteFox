
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 8, 16, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv1d(8, 8, 1, stride=1, padding=0, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 512)
x2 = torch.randn(1, 2, 512)
