
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 257)
        self.linear2 = torch.nn.Linear(257, 512)
        self.linear3 = torch.nn.Linear(512, 801)
        self.linear4 = torch.nn.Linear(801, 2)
        self.__in_features = 2
    def forward(self, x1):
        v1 = x1
        v2 = self.linear1(v1)
        v4 = self.linear2(v2)
        v6 = self.linear3(v4)
        v7 = torch.nn.functional.relu(v6)
        v9 = torch.nn.functional.linear(v7, self.linear4.weight, self.linear4.bias)
        v11 = v9.transpose(1, 2)
        return v11
# Inputs to the model
x1 = torch.randn(1, 2, 2)
