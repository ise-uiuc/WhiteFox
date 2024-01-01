
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.Identity()
    def forward(self, x):
        v1 = torch.tanh(x)
        v2 = self.op(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 3, 4, 4)
