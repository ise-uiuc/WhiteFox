
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_20 = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        block_21 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        block_22 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True)]
        self.features = torch.nn.Sequential(*block_20, *block_21, *block_22)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
