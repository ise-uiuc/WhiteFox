
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(25, 100, 1)
        self.linear = torch.nn.Linear(100*2, 5)
    def forward(self, x):
        v1 = self.conv(x)
        v1 = torch.flatten(v1, 1)
        v1 = torch.tanh(v1)
        v1 = torch.nn.functional.dropout(v1)
        v1 = torch.relu(v1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.sigmoid(v2)
        return v2

