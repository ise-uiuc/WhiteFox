
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.permute = Permute()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.permute(x1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        y = self.sigmoid(v2)
        w = self.relu(y)
        return w
# Inputs to the model
x1 = torch.randn(1, 2, 2)
