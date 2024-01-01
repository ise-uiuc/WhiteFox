
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        if self.linear.bias is None:
            raise AssertionError('Bias is not used in linear layer')
        else:
            v1 = x1.permute(0, 2, 1)
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            x2 = torch.nn.functional.relu(v2)
            v3 = x2.detach()
            v4 = torch.min(v3, dim=-1)[1]
            return v4
# Inputs to the model
x1 = torch.randn(1, 2, 4)  # Input shape is (1, 2, 4)
