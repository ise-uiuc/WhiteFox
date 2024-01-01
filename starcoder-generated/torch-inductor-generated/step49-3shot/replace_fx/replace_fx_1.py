
x1 = torch.randn(2, 3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return x1.new_empty(x1.shape).normal_()
# Inputs to the model
# Inputs to the model
x1 = torch.randn(2, 3)
