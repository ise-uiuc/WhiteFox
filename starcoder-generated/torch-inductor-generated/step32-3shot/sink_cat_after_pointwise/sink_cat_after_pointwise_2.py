
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = torch.cat((x, x), dim=0)
        y = z.view(-1)
        return torch.relu(y) if y.shape!= (1, ) else torch.relu(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
