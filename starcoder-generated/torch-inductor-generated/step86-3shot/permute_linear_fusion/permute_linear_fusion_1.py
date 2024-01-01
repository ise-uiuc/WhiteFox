
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
    def forward(self, x1):
        v1 = torch.flatten(x1)
        v2 = torch.cat((v1, v1), 0)
        v3 = v2.permute(2, 0, 1)
        v4 = torch.nn.functional.linear(v3, torch.relu(self.linear.weight), self.linear.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
