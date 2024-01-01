
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1
        v3 = torch.tensor(1)
        v2 = v1.permute(0, 2, v1.size()[0], v1.size()[1])
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
