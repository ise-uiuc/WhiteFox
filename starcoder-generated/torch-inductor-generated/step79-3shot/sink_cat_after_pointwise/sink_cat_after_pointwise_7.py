
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
    def forward(self, x):
        y = self.linear(x)
        z = torch.cat((y, y), dim=1)
        z1 = z * 5
        z2 = self.relu(z1)
        if self.training and len(self.linear._parameters) > 0:
            z0 = y * 3
            z0 = self.tanh(z0)
        z4 = self.tanh(z0)
        z = z2 * 7 + z4
        return z 
# Inputs to the model
x1 = torch.randn(1, 3)
