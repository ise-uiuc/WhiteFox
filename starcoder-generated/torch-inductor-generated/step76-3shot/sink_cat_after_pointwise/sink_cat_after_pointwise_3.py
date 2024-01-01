
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(3):
            z = torch.cat([y, y, y], dim=1)
            y = z.tanh()
        return torch.relu(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
