
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x[0:1, :, :] # slicing
        x = y.relu()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
