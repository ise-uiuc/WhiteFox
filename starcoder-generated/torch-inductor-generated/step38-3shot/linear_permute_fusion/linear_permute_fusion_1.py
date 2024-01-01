
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1
        for i in range(10):
            v2 = torch.nn.functional.relu(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias))
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2).cuda()
