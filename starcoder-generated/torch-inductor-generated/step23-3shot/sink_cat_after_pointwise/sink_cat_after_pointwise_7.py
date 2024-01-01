
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(5, 4)
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = torch.matmul(x, self.weight)
        x = x.view(-1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
