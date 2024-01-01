
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(3, 80).permute(1, 0)
        y = y.tanh()
        y = y.t()
        y = y.tanh()
        y = y.permute(1, 0)
        y = y.view(3, 80)
        z = torch.cat((y, y), dim=0)
        if z.shape[0] == 2:
            z = z.tanh()
        else:
            z = z.reshape(2,10)
        z = torch.relu(z)
        x = torch.cat((z,z), dim=2)
        y = x.tanh()
        z = y.view(-1).tanh()
        return z
# Inputs to the model
x = torch.randn(2, 3, 10)
