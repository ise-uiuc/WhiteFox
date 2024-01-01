
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=1)
        v2 = v1.shape[0]
        v3 = torch.relu(v2)
        return v3.view(-1)
# Inputs to the model
x1 = torch.randn(1, 2)
