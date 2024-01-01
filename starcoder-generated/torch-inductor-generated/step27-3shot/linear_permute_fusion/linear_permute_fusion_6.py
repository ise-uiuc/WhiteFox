
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1, 3)
        y = (v2 * self.linear2.weight.unsqueeze(0)).sum([0, 1, 2])
        return v1
# Inputs to the model
x1 = torch.randn(2, 2, 3, device='cpu')
