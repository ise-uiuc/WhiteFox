 from PyTorch:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
        )
        self.split = torch.nn.MaxPool2d(3, 2, 1, 1)
        self.split1 = torch.nn.MaxPool2d(3, 1)
        self.split2 = torch.nn.MaxPool2d(5, 4)
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 1], 1)
        concatenated_tensor = torch.cat(split_tensors, 1)
        v3, _ = self.split(v1)
        _, split_tensors_1 = self.split1(v3)
        _, split_tensors2 = self.split2(v3)

        return (concatenated_tensor, split_tensors)
# Inputs to the model ends
