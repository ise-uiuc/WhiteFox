
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.reshape(v2.shape[0], v2.shape[1], 2, 2)
        v4 = v3.permute(0, 3, 2, 1)
        return v4.reshape(v4.shape[0], v4.shape[1], v2.shape[2])
# Inputs to the model
x1 = torch.randn(1, 2, 2)
