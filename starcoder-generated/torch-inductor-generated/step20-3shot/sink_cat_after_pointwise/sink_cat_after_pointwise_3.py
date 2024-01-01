
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 2)
        self.bn1 = torch.nn.BatchNorm1d(num_features=3, affine=False)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.linear2 = torch.nn.Linear(3, 2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=3, affine=False)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.bn1(x)
        a = self.linear2(l1 + l2)
        a = self.bn2(a)
        a = self.relu1(a)
        x = self.bn2(torch.relu(self.linear2(self.bn1(self.linear1(x)) + a)))
        x = self.sigmoid1(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
