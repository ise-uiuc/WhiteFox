
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv1d(3, 3, 1), torch.nn.BatchNorm1d(3, affine=False))
        torch.manual_seed(1)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.relu = torch.nn.ReLU()
        torch.manual_seed(2)
        self.pooling = torch.nn.AvgPool1d(3, ceil_mode=False)
    def forward(self, x1):
        y1 = self.block1(x1)
        y1 = self.dropout(y1)
        y1 = self.relu(y1)
        y1 = self.pooling(y1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 3, 10)
