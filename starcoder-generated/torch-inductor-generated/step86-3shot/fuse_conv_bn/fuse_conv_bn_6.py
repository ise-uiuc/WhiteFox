
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv1d(5, 3, 3, padding=1, dilation=2, bias=True)
        bn = torch.nn.BatchNorm1d(3)
        self.layer = torch.nn.Sequential(conv, bn)
    def forward(self, x):
        x = self.layer(x)
        return x
# Inputs to the model
x = torch.randn(1, 5, 3)
