
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 2)
        self.linear2 = torch.nn.Linear(2, 4)
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous().view(-1, 4)
        x = torch.tanh(self.linear1(x))
        return self.linear2(x).view(x.size(0), 2, -1).permute(0, 2, 1)
# Inputs to the model
x = torch.randn(2, 2, 4)
