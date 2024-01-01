
class Model(torch.nn.Module):
    def add_scalar(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = x2.clone()
        o1 = v2 + x2
        return o1

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = self.add_scalar(v1, x1)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.max(x2, dim=1)[-1]
        v3 = v3.unsqueeze(-1)
        return v3

    def forward_add_scalar(self, x1):
        v1 = self.add_scalar(x1, x1)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.max(v3, dim=1)[-1]
        v4 = v4.unsqueeze(-1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
