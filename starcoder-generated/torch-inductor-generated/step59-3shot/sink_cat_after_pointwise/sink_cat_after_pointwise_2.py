
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1).to(torch.bfloat16)
        return torch.relu(y.view(y.shape[0], -1), inplace=False)
# Inputs to the model
x = torch.randn(2, 3, 4)
