
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.permute(1, 2, 0)
        z = y.mean(dim=3, keepdim=True)
        u = y if x.shape[0] else z
        return (u + z).transpose(1, 2).squeeze(3).mean(dim=(0, 1)).sum()
# Inputs to the model
x = torch.randn(5, 1, 12, 256)
