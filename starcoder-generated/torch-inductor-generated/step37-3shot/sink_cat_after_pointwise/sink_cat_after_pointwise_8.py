
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = torch.relu(x)
        x = torch.cat((x, x), dim=1)
        x = x.view(x.shape[0], -1).unsqueeze(-1)
        x = torch.mean(x, dim=1, keepdim=False)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
