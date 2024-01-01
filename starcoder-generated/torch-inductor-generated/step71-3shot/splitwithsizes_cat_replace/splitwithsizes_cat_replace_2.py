
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleDict({'conv1': torch.nn.Conv2d(16, 2, (2, 2)), 'conv2': torch.nn.Conv2d(2, 4, (2, 2))})
    def forward(self, v1):
        x = torch.transpose(v1, 1, 2)
        split_tensors = torch.split(x, [3, 4], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return torch.transpose(concatenated_tensor, 1, 2)
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)
