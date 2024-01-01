
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.mm(x1, x2), torch.unsqueeze(torch.mm(x2, x1), 0), torch.unsqueeze(torch.mm(x1, x2), 0), torch.mm(x1, x2)], 0)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 2)
