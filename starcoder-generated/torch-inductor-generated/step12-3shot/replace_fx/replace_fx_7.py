
class Model(torch.nn.Module):
    def __init__(self, d=0.5):
        super().__init__()
        self.d = d
    def forward(self, input_tensor):
        x = torch.rand_like(input_tensor)
        return x > self.d
# Inputs to the model
x1 = torch.randn((3, 4))
x2 = torch.randn((5, 3, 4))
x3 = torch.randn((7, 7, 4))
x4 = torch.randn((8, 3, 2, 4))
