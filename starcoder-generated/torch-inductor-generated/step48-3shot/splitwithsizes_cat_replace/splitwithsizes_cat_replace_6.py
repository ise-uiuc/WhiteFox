
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 4), torch.nn.Flatten(),])
    def forward(self, v1):
        (x1, x2, x3) = tensor_tuple
        return (x2, tensor_tuple)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 32, 60, 60)
x3 = torch.randn(1, 32, 30, 30)
tensor_tuple = (x1, x2, x3)
