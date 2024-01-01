
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(1, 20, 5, 1, padding=2, bias=False)]
        block_1 = [torch.nn.MaxPool2d(2)]
        block_2 = [torch.nn.Conv2d(20, 50, 5, 1, bias=False)]
        block_3 = [torch.nn.MaxPool2d(2)]
        block_4 = [torch.nn.Conv2d(50, 500, 4, 1, padding=1, bias=False)]
        block_5 = [torch.nn.MaxPool2d(2)]
        block_6 = [torch.nn.Linear(800, 500, bias=False)]
        block_7 = [torch.nn.MaxPool2d(2)]
        block_8 = [torch.nn.Linear(8400, 10, bias=True)]
        block_9 = [torch.nn.ReLU()]
        block_10 = [torch.nn.Linear(500, 10, bias=True)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7, *block_8, *block_9, *block_10)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
