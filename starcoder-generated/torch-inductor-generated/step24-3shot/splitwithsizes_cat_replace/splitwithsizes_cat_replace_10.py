
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 32, 3, 1, 1))
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 1, 0))
        self.features_1 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 1, 2))
        self.features_2 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 1, 2), torch.nn.ReLU(inplace=True))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        intermediate, _ = self.features_1(split_tensors[1])
        intermediate, _ = self.features_2(intermediate)
        _, split = self.features(split_tensors[1])
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1), intermediate, split)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
