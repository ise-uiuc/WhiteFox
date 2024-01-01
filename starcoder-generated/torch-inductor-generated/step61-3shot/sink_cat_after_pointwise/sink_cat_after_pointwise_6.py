
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y):
        x1 = torch.cat([y.view(-1, *y.shape[2:])] * 3, dim=2)
        x1 = x1.tanh()
        x1 = x1.view(x1.shape[0] // 3, 3, *x1.shape[1:]).permute(2, 0, 1, 3)
        x1 = torch.relu(x1)
        return x1
# Inputs to the model
y = torch.randn(2, 4, 2, 2)
