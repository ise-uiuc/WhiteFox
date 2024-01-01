
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 7)
    def forward(self, x):
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        x_new = x[:, :, 1:].contiguous().permute(1, 2, 0).contiguous()
        x = torch.cat((x, x_new), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 3)
