
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(6)
        self.layer = torch.nn.Sequential( torch.nn.Conv1d(6, 7, 2, bias=False), torch.nn.BatchNorm1d(7), torch.nn.ReLU6())
    def forward(self, x1):
        s1 = self.layer(x1)
        return s1 + s1
# Inputs to the model
x1 = torch.randn(1, 6, 8)
