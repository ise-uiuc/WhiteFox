
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.cat((x1, 1e5*torch.randn_like(-x1)), dim=0)
# Inputs to the model
x1 = torch.randn(5,5, device='cuda:0')
a = []
for i in range(20):
    a.append(x1)
x1 = a
