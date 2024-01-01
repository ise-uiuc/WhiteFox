
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.linear2(v1)
        v5 = v4.squeeze(1)
        return v3 + v5
# Inputs to the model
x1 = torch.randn(1, 2, 3)
