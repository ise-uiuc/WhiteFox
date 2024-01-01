
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Flatten([2, 3])
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 5, 2)
