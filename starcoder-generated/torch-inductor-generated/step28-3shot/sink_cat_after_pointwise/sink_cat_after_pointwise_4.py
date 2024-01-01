
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.layer_norm = torch.nn.LayerNorm((4,))
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = self.fc1(y)
        y = self.layer_norm(y)
        x = y.view(-1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
