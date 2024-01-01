2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU(15)
    def forward(self, x1):
        return self.prelu(x1)
# Input to the model is x_input
x_input = torch.randn(1, 15, 4, 6)
