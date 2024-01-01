
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 5, 1, 2), torch.nn.Conv2d(32, 32, 3, 1, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=3)
        concatenated_tensor = torch.cat(split_tensors, dim=3)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=3))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
