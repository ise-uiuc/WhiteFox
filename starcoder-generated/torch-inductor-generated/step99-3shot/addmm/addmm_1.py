
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + x1
        return v1
# Inputs to the model, also used as target
x1 = torch.randn(2, 3)
