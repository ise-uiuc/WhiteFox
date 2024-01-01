
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 1, 1, 0)])
    def forward(self, v1):
        split_tensor = torch.split(v1, [1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensor, dim=1)
        return (concatenated_tensor,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
