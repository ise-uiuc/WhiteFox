
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x):
        a = x.shape
        b = a[-1]
        v1 = x.new_zeros(a[:-1] + (b - 1,))
        v2 = torch.cat((x, v1), dim=-1)
        y = torch.stack((v1), dim=-1).flatten()
        return torch.nn.functional.relu(y)
# Inputs to the model
x = torch.randn(1, 5, 3)
