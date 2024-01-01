
x = torch.randn(3, 10) + torch.randn(10)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y):
        return y+x
# Input to the model
y = torch.randn(3, 10)
