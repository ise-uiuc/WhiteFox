
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(1, 1, 1, 1, 0, bias=False),]
        self.features = torch.nn.Sequential(*block * 2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
