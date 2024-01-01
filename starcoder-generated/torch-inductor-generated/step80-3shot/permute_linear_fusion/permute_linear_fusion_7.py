
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v3 = torch.nn.functional.relu(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias))
        v2 = torch.max(v3, dim=-1)[0]
        v3 = torch.nn.functional.relu(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias))
        x2 = torch.nn.functional.softmax(v2)
        return torch.sum(v3) + torch.sum(x2.to(v3.dtype))
# Inputs to the model
x1 = torch.randn(1, 3, 3)
