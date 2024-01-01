
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        block_1 = [torch.nn.ReLU()]
        block_2 = [torch.nn.MaxPool2d(3, 1, 1)]
        block_3 = [torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)]
        block_4 = [torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)]
        block_5 = [torch.nn.ReLU()]
        block_6 = [torch.nn.AvgPool2d(3, 1, 1)]
        block_7 = [torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = [(split_tensors[i][:] * 2).detach() for i in range(len(split_tensors))]
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
