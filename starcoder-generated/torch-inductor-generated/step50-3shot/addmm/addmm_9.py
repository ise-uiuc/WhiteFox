
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.randn(3, 3, requires_grad=True)
    def forward(self, x):
        v1 = torch.mm(x, torch.tanh(torch.tanh(torch.mm(x, self.input))))
        return v1
# Inputs to the model
x = torch.randn(5, 5)
