
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.leaky_relu = torch.nn.LeakyReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight)
        x2 = self.leaky_relu(v1)
        z = (x1 * 2) ** self.leaky_relu(v2)
        return z + x1 + x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
