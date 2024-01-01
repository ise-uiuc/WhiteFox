
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.parameter.Parameter are trainable tensors
        self.w = torch.nn.Parameter(torch.randn(3, 3))
    def forward(self, x1, x2):
        return torch.add(torch.mm(x1, x2), torch.mm(self.w, self.w))
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
