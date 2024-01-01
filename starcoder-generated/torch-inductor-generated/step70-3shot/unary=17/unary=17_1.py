
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU6()
    def forward(self, x1):
        return self.activation(torch.nn.Conv2d(3, 16, 3)(x1))
# Input to the model
x1 = torch.randn(1, 3, 112, 96)
