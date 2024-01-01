
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.reshape(-1, 2)
        v4 = self.relu(v3)
        v4 = v4.reshape(x1.shape)
        v4 = v2 - v4
        v3 = v1 - v4
        v3 = self.softmax(v3)
        v3 = v1 - v3
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
