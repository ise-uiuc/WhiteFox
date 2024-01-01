
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v0.relu()
        v3 = v1.unsqueeze(0)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v2 = v1.softmax(dim=0)
        return v4
# Inputs to the model
x0 = torch.randn(1, 2, 2)
