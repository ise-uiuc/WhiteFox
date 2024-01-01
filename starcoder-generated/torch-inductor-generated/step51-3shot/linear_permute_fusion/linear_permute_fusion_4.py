
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = torch.zeros(4, 2, 2)
        x3 = x2.permute(0, 2, 1)
        v2 = v1.permute(0, 2, 1)
        x4 = torch.zeros(4, 2, 2)
        v2 = x4.permute(0, 2, 1)
        return torch.nn.functional.linear(x4, torch.tensor([[-1.0672, 0.9924],
 [0.6685, 0.6025]], requires_grad=True), v2)
# Inputs to the model
x1 = torch.randn(4, 2, 2)
