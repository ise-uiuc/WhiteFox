
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, input):
        v1 = input + 2.0
        v2 = v1.permute(0, 2, 1)
        v2 = v2 / 1.5
        v3 = v2 + 0.2
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 3)
