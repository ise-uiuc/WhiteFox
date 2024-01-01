
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        block_0 = [torch.nn.Conv2d(3, 64, 3, 1, 0, bias=False, stride=2)]
        block_1 = [torch.nn.BatchNorm2d(64), torch.nn.Dropout(0.25), torch.nn.ReLU()]
        block_2 = [torch.nn.Conv2d(64, 128, 3, 1, 0, bias=False, stride=2)]
        block_3 = [torch.nn.MaxPool2d(3, 2)]
        block_4 = [torch.nn.BatchNorm2d(128), torch.nn.Dropout(0.25), torch.nn.ReLU()]
        block_5 = [torch.nn.Conv2d(128, 128, 3, 1, 0, bias=False, stride=1)]
        block_6 = [torch.nn.BatchNorm2d(128), torch.nn.Dropout(0.4)]
        block_7 = [torch.nn.AvgPool2d(6, 3, 2), torch.nn.Dropout(0.4)]
        block_8 = [torch.nn.Conv2d(128, 128, 1, 1, 0, bias=False)]
        block_9 = [torch.nn.BatchNorm2d(128), torch.nn.Dropout(0.4)]
        block_10 = [torch.nn.Flatten()]
        block_11 = [torch.nn.Linear(3136, 1000)]
        block_12 = [torch.nn.ReLU()]
        block_13 = [torch.nn.Linear(1000, 1000)]
        block_14 = [torch.nn.ReLU()]
        block_15 = [torch.nn.Linear(1000, 10)]
        self.block = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7, *block_8, *block_9, *block_10, *block_11, *block_12, *block_13, *block_14, *block_15)
    def forward(self, v1):
        v35, v36 = self.features(v1)
        v43, v44 = torch.split(v35, [2048, 2048], 1)
        v45 = v36 + v43
        return (torch.split(v45, [2048, 2048], 1), torch.split(v35, [2048, 2048], 1))
# Inputs to the model
x1 = torch.randn(10, 3, 512, 512)
