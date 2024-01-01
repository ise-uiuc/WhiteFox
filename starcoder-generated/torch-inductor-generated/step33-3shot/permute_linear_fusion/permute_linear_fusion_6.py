
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        y = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        y1 = y.flip(0)
        y2 = y.clone()
        y2[0][0][1] = 2.3
        return y1 - y2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
