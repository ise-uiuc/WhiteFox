
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear_0 = [torch.nn.BatchNorm2d(3)]
        linear_1 = [torch.nn.Linear(64, 64)]
        self.features = torch.nn.Sequential(*linear_0, *linear_1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
