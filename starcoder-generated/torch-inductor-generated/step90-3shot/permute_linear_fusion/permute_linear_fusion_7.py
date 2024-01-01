
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(349, 1043)
        self.linear2 = torch.nn.Linear(6, 4)
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        return v2 + v3
# Inputs to the model
x = torch.randn(1, 6, 349)
