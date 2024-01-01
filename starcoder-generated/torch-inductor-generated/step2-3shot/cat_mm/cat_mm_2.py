, changed the model to return a list of tensor.
# The length of the list is random.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return [torch.add(x, x), torch.subtract(x, x),]
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 1)
