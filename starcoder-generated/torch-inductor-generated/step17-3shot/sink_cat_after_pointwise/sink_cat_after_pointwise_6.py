
class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x, y):
        z0 = torch.cat((x, y), dim=1)
        out = self.relu(z0)
        out = out.view(-1)
        out = self.relu(out)
        out = out.view(-1)
        return out
class ModelB(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z0 = torch.cat((x, y), dim=1)
        z1 = z0.permute((1, 0, 2)).reshape(z0.shape[0], -1)
        z2 = torch.relu(z1)
        z3 = z2.view(z2.shape[0], 1, z2.shape[1]).permute(1, 0, 2)
        out = torch.cat((z3, z3), dim=1)
        return out
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
model_A = ModelA()
model_B = ModelB()
torch.jit.trace(model_A, (x, y))
torch.jit.trace(model_B, (x, y))
