
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.sigmoid(v2).clone()
        x2 = torch.nn.functional.relu(x2).clone()
        v3 = torch.nn.functional.max_pool2d(x2.clone(), 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
