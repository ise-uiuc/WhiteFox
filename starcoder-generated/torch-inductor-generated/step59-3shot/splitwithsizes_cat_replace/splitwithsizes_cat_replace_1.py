
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, 1, 1), torch.nn.ReLU(inplace=False))
        self.features_1 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, 1, 1), torch.nn.ReLU(inplace=False))
        self.features_2 = torch.nn.Sequential(torch.nn.Conv2d(32, 48, 3, 4, 1), torch.nn.ReLU(inplace=False))
        self.features_3 = torch.nn.Sequential(torch.nn.Conv2d(48, 64, 3, 1, 3), torch.nn.ReLU(inplace=False))
        self.maxpool = torch.nn.MaxPool2d(2)
        self.features_4 = torch.nn.Sequential(torch.nn.Conv2d(64, 80, 3, 1, 0), torch.nn.ReLU(inplace=False))
        self.features_5 = torch.nn.Sequential(torch.nn.Conv2d(80, 96, 3, 1, 0), torch.nn.ReLU(inplace=False))
        self.features_6 = torch.nn.Sequential(torch.nn.Conv2d(96, 128, 3, 1, 0), torch.nn.ReLU(inplace=False), torch.nn.AvgPool2d(kernel_size=2))
    def forward(self, v1):
        split_tensors = torch.split(v1, [13, 3, 120, 16, 84, 48], dim=1)
        input = [split_tensors[0], split_tensors[2], split_tensors[3], split_tensors[5]]
        concatenated_tensor = torch.cat(input, dim=1)
        f1, f2, f3, f4, f5, f6 = torch.split(concatenated_tensor, split_sizes=[13, 3, 120, 16, 84, 48], dim=1)
        return (f1, [f2, f3, f4, f5, f6])
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
