
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.cat((x1, x2), dim=1)
        x3 = torch.relu(x3)
        return x3 if x1.shape[0] == 1 else x3.view(-1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 3)
