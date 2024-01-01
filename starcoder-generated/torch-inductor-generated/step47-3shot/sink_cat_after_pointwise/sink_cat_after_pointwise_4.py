
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        x: torch.Tensor,
    ):
        x = torch.cat((x, x, x), dim=1)
        # no need to sink cat since x has shape (1, 3)
        if x.shape!= (1, 3):
            x = x.view(x.shape[0], -1)
        return x.relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
