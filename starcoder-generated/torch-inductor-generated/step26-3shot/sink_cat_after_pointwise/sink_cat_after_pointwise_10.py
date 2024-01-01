
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        # concat1
        x1 = x1.view(x1.shape[0], -1)
        x1 = x1.contiguous()
        x1 = x1.cat([x1, x1], dim=-1)
        # concat2
        x1 = torch.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
