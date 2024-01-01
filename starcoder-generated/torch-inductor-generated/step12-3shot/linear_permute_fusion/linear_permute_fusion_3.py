
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.clone()
        v2 = v1.permute(0, 2, 1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 2, 2)
