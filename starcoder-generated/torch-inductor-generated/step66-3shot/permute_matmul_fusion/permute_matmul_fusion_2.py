
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1))[0][0][0], torch.bmm(x2, x1.unsqueeze(2))[0][0][0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
