
class Model1(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReplicationPad2d(4), torch.nn.Conv2d(3, num_features, 8, 4, 0, bias=False), torch.nn.BatchNorm2d(num_features), torch.nn.ReLU(), torch.nn.Conv2d(num_features, num_features, 4, 2, 1, bias=False), torch.nn.BatchNorm2d(num_features), torch.nn.ReLU(), torch.nn.Conv2d(num_features, num_features, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(num_features), torch.nn.ReLU())
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v0, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = torch.nn.Sequential(Model1(num_features))
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v0, [1, 1, 1], dim=1))
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
