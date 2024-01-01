
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(73, 87)
    def forward(self, x):
        x6 = self.linear_0(x)
        v1 = x6 > 0
        v2 = x6 * -4.75
        v3 = torch.where(v1, x6, v2)
        return v3
# Inputs to the model
x = torch.randint(53, (11, 73))

y = torch.randint(561, (73,))
