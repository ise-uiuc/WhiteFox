
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.test_1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.test_2 = nn.Conv1d(3, 3, 3, 1, 1)

    def forward(self, x):
        x_r, x_c = torch.split(x, 3, dim=-1)
        x = self.test_1(x)
        x_r, x_c = x_r + x_c, torch.cat([self.test_2(x).squeeze(-1), x_c], dim=1)
        x_cat = torch.cat([x_r, x_c], dim=-1)
        return x_cat, x
# Inputs to the model
x = torch.randn(4, 3, 8, 8)
