
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(1, 32, 3, 1, 1, bias=True)]
        block_1 = [torch.nn.ReLU()]
        block_2 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True)]
        block_3 = [torch.nn.BatchNorm2d(32)]
        block_4 = [torch.nn.ReLU()]
        block_5 = [torch.nn.Conv2d(32, 64, 3, 1, 0, bias=True), torch.nn.BatchNorm2d(64)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
