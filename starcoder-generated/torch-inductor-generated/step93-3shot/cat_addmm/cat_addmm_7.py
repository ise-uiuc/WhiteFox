
class Model(nn.Module):
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = torch.stack((x, x, x, x, x), dim=1)
        x = torch.stack((x, x, x, x, x), dim=1)
        x = torch.stack((x, x, x, x, x), dim=2)
        x = x.flatten(end_dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 2)
