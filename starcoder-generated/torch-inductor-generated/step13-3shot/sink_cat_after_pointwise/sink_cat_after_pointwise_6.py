
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat([x, x], dim=1)
        t2 = torch.relu(t1)
        y = t2.reshape(-1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
