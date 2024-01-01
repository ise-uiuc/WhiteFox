
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 2, 2, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x):
        v1 = torch.nn.functional.dropout(x, p=0.0)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
