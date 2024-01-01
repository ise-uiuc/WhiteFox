
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
        self.sigmoid = torch.sigmoid
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v0.reshape(3, 2)
        v2 = torch.bmm(torch.tanh(v1), self.linear2.weight.permute(1, 0))
        v3 = torch.sigmoid(v2).permute(1, 0)
        return v3
# Inputs to the model
x0 = torch.randn(1, 2)
