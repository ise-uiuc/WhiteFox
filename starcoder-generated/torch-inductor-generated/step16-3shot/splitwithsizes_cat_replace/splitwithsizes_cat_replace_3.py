
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.cat = torch.nn.Sequential(torch.nn.Conv2d(6, 32, 3, 1, 1))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        first_tensor = split_tensors[0]
        second_tensor = split_tensors[1]
        third_tensor = split_tensors[2]
        concatenated_tensor = torch.cat([first_tensor, second_tensor, third_tensor], dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
