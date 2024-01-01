
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.conv = torch.nn.Conv2d(self.in_features, self.out_features, 1)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        x.clone().detach().requires_grad_(True) 
        x = self.conv(x)
        x = torch.cat((x, x), dim=-1).tanh()
        return x
# Inputs to the model
x = torch.randn(1, 2, 32, 32)
