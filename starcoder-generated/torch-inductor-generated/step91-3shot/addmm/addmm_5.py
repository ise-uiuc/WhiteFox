
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(16, 16, bias=True)
        self.w = torch.norm(self.model.weight, dim=1) + torch.norm(self.model.bias)
    def forward(self, inp):
        self.model.weight = 10*torch.eye(16)
        self.model.bias = torch.zeros(16)
        temp = self.model(inp)
        return torch.mm(temp, inp) / self.w
# Inputs to the model
inp = torch.randn(16, 16, requires_grad=True)
