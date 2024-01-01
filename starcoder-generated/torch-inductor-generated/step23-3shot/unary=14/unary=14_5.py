
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_12 = torch.nn.Softmax(0)
    def forward(self, x1):
        v1 = self.softmax_12(x1)
        return v1
# Inputs to the model
x1 = torch.clamp(torch.rand((2, 6)), min=0, max=1)
