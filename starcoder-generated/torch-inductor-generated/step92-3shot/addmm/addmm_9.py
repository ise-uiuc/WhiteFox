
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = 2 * torch.randn(3, 3, requires_grad=True)
    def forward(self, x1):
        v1 = torch.mm(x1, x1)
        self.x = v1 * self.x
        return "success"
# Inputs to the model
x1 = torch.zeros(3, 3)
