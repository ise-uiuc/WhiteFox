
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv1d(3, 3, 1), torch.nn.Conv1d(3, 3, 1))
        self.split = torch.nn.Sequential(torch.nn.AvgPool2d(2, 1, 1), torch.nn.AvgPool2d(3, 2, 1), torch.nn.AvgPool2d(2, 1, 1))
    def forward(self, x0):
        v0 = self.features(x0)
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (v0, concatenated_tensor, torch.split(v0, [1, 1, 1], dim=1))
# Inputs to the model
x0 = torch.randn(3, 3, 64)
