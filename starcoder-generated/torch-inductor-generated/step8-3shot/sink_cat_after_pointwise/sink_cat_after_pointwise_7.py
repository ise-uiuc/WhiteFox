
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # torch.cat() is done with the user-input dimension.
        y = torch.stack(
            [x, torch.add(x, torch.ones_like(x))],
            dim=1
        )
        # torch.add() is done with the user-input dimension.
        x = x.view(x.shape[0], -1).tanh() if x.shape[0] == 1 else x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
