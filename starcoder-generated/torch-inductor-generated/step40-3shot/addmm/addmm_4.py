
class Model(torch.nn.Module):
    def __init__(self, requires_grad):
        super().__init__()
        self.x = torch.randn(3, requires_grad=requires_grad)
    def forward(self, y):
        return y + self.x
# Inputs to the model
y = torch.randn(3, requires_grad=True)
