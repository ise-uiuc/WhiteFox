
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 8)
    def forward(self, x1):
        v1 = x1 - 5.0
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x1 = torch.nn.functional.relu(v2)
        x1 = x1.type(torch.double)
        v3 = torch.max(x1, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        x2 = torch.cat((v3, v4), dim=-1)
        x2 = self.linear2(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2)
