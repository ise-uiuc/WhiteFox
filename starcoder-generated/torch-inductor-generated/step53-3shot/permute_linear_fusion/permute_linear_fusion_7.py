
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(576, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.clone()
        v4 = v2 + x2
        v4 = v4.unsqueeze(dim=1).clone()
        v6 = v4 * v3
        v6 = torch.sum(v6, dim=2)
        v4 = v4 - v6
        return (v3!= 1.0) | (v4 > 0.0)
# Inputs to the model
x1 = torch.randn(2, 2, 576)
