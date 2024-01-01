
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
    def forward(self, x):
        x1 = x.permute(0, 2, 1)
        x1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        x1 = torch.nn.functional.relu(x1)
        x2 = torch.max(x1, dim=(-2, -1))[0]
        x2 = x2.unsqueeze(dim=-1)
        x2 = x2.unsqueeze(dim=-1)
        x3 = x1 - x2
        x3 = torch.mean(x3, dim=(-2, -1))
        x3 = x3.reshape((-1, 2, 4))
        x4 = torch.max(x3, dim=-1)[0]
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
