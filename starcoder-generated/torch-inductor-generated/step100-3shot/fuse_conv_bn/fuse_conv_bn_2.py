
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.linear1 = torch.nn.Linear(20, 10)
        torch.manual_seed(3)
        self.linear1_bn = torch.nn.BatchNorm1d(10)
        self.relu = torch.nn.ReLU()
        torch.manual_seed(2)
        self.linear2 = torch.nn.Linear(10, 10)
        torch.manual_seed(1)
        self.linear2_bn = torch.nn.BatchNorm1d(10)
    def forward(self, x1):
        y1 = self.linear1(x1)
        y1 = self.linear1_bn(y1)
        y1 = self.relu(y1)
        y1 = self.linear2(y1)
        y1 = self.linear2_bn(y1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 20)
