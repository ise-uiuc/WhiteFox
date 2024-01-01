
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Linear(2, 1)
        self.t2 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1.transpose(2, 1))
        v2 = torch.nn.functional.linear(v1, self.t1.weight, self.t1.bias)
        v3 = v2.transpose(1, 2)
        v4 = torch.nn.functional.linear(v2, self.t2.weight, self.t2.bias)
        return v4.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
