
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Conv1D(5, 5, 3), torch.nn.BatchNorm1D(5)) # noqa: E501
    def forward(self, x0):
        y = self.layer(x0)
        return y + y
# Inputs to the model
x0 = torch.randn(1, 5, 7)
