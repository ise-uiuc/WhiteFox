
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.ReLU(inplace=False), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 8))
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1])
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
