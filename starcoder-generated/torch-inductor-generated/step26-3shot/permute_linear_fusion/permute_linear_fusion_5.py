
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(480, 800)
        self.relu = torch.nn.ReLU()
        self.max = torch.nn.MaxPool1d(13)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.max(self.relu(v2))
        return v1
# Inputs to the model
x1 = torch.randn(1, 480, 2)
