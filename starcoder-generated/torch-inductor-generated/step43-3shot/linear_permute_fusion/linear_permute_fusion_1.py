
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x2):
        z0 = x2
        w2 = torch.nn.functional.linear(z0, self.linear.weight, self.linear.bias)
        y0 = w2.permute(0, 2, 1)
        y1 = y0.reshape(-1)
        return y1
# Inputs to the model
x2 = torch.randn(1, 3, 3)
