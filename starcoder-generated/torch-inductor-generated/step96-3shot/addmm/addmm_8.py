
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(3, 3, bias=False)
        self.m.weight.data = torch.ones(3, 3) * -1
    def forward(self, tensor):
        return tensor + self.m(tensor)
# Inputs to the model
tensor = torch.randn(3, 3, requires_grad=True)
