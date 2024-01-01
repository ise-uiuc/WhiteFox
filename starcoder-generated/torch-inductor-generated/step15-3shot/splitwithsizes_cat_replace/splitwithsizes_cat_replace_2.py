
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.Sequential(torch.nn.Conv2d(3, 10, 3, 1, 1), torch.nn.Dropout2d())
        self.features = torch.nn.Sequential(torch.nn.Conv2d(10, 10, 3, 1, 1), torch.nn.Conv2d(10, 10, 3, 2, 3), torch.nn.Conv2d(10, 10, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(10, 10, 3, 2, 3))
    def forward(self, x):
        v1 = self.convs(x)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
