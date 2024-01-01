
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.unsqueeze(torch.unsqueeze(torch.mm(x1, x2), 0), 0), torch.unsqueeze(torch.unsqueeze(torch.mm(x1, x2), 0), 0), ], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
