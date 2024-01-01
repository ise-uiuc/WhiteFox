
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.Tanh()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.softmax(v2)
        v4 = v3.unsqueeze(1)
        v5 = torch.nn.functional.linear(v4, self.linear.weight.transpose(-2, -1), self.linear.bias)
        v6 = v4 + v5
        v7 = self.activation(v6)
        v8 = v7.squeeze(1)
        v9 = v1 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 2, 2)
