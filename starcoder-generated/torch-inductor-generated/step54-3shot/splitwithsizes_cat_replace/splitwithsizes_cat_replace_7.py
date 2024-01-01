
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1x5, 3x5, 3x3, 1x3
        self.features = torch.nn.ModuleList([torch.nn.AvgPool2d(1, 2, 0), torch.nn.AvgPool2d(3, 2, 0), torch.nn.AvgPool2d(3, 1, 0), torch.nn.AvgPool2d(1, 1, 0)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
