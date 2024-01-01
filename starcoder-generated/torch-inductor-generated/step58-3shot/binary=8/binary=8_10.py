
# Note: For this model, only the forward function are needed.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, 8)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x1, x2):
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x1 = self.relu1(self.linear1(x1))
        x2 = self.relu2(self.linear2(x2))
        y = self.softmax(1.0 * x1.mul(x2))
        return y

x1 = torch.randn(1, 16, 2, 2)
x2 = torch.randn(1, 16, 2, 2)
