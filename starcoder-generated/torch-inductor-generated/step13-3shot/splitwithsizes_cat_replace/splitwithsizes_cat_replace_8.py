
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), torch.nn.Conv2d(3, 12, 3, 1, 1))
        self.split_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 1, 3, 1, 1))
        self.split_2 = torch.nn.Sequential(torch.nn.Conv2d(12, 1, 3, 1, 1))
        self.split_3 = torch.nn.Sequential(torch.nn.Conv2d(3, 1, 3, 1, 1))
        self.split_4 = torch.nn.Sequential(torch.nn.Conv2d(12, 1, 3, 1, 1))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors_1 = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors_2 = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors_3 = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors_4 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors_1, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
