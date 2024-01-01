
class CustomModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(CustomModule, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.linear(x, torch.ones(4, 4, device=x.device), self.inplace)
model = CustomModule(inplace=False)
# Inputs to the model
x = torch.randn(2, 4, 4)
