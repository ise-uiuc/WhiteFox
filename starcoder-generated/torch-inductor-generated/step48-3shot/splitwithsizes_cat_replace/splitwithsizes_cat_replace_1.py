
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1)])
    def forward(self, v2):
        split_tensors = torch.split(v2, [1, 1, 1], dim=3)
        concatenated_tensor = torch.cat(split_tensors, dim=3)
        return (concatenated_tensor, torch.split(v2, [1, 1, 1], dim=3))
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
