
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer = torch.nn.Sequential(torch.nn.Conv1d(3, 3, 3, groups = 3, bias=True), torch.nn.BatchNorm1d(3))
    def forward(self, x2):
        s2 = self.layer(x2)
        return s2 + s2
# Inputs to the model
x2 = torch.randn(1, 3, 3)
