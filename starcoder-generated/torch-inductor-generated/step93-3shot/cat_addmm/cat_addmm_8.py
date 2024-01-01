
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 3)
        self.layers_2 = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = x.flatten(start_dim=1, end_dim=2)
        # x = x.permute(0, 2, 1) # Uncomment this line to trigger the bug
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
