
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.concat = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 2, 1)), torch.nn.AdaptiveAvgPool2d((1, 2, 1)), torch.nn.AdaptiveAvgPool2d((1, 2, 1)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
