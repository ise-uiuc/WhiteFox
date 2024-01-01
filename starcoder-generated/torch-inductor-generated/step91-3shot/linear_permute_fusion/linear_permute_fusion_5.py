
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 3)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(3, 4)
        self.tanh = torch.nn.Tanh()
        self.linear_3 = torch.nn.Linear(4, 2)
    def forward(self, x1):
        v0 = x1
        v1 = torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias)
        v2 = self.relu(v1)
        v3 = torch.nn.functional.linear(v2, self.linear_2.weight, self.linear_2.bias)
        v4 = self.tanh(v3)
        v5 = torch.nn.functional.linear(v4, self.linear_3.weight, self.linear_3.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
