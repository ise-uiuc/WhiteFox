
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ = torch.nn.modules.conv.Conv1d(2, 2, 2)
    def forward(self, x1):
        v1 = self.conv_(x1)
        v2 = v1.permute(0, 1, 3, 2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
