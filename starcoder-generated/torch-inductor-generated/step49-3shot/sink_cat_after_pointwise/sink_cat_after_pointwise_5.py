
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10).float()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(-1)
        if x.shape[0]!= 100:
            x = self.linear(x.unsqueeze(0))
        else:
            x = self.linear(x)
        return x
# Inputs to the model
x = torch.randn(100, 5)
