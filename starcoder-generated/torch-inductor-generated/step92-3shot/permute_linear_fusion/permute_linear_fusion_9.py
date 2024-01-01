
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout()
    def forward(self, x):
        v = torch.transpose(x, -1, -2)
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v2 = torch.matmul(v, v1)
        v3 = self.softmax(v2)
        v4 = self.ReLU(v3)
        v5 = self.dropout(v1)
        v6 = torch.matmul(v4, v5)
        return v6
# Inputs to the model
x = torch.randn(1, 10)
