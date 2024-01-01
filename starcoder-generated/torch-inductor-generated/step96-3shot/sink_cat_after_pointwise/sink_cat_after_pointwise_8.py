
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 4)
    def forward(self, x):
        y = x
        x = y + self.linear(y.view(-1, 128)).view(-1)
        z = torch.transpose(x.view(x.shape[0], -1), 0, 1).contiguous()
        x = torch.cumsum(z, dim=1)
        return x
# Input to model
shape = torch.tensor((2, 128))
