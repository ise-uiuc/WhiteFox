
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 2, 3), torch.nn.Conv2d(32, 64, 3, stride=1, padding=1), torch.nn.Conv2d(64, 64, 3, 2, 3)])
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 2, 3))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
