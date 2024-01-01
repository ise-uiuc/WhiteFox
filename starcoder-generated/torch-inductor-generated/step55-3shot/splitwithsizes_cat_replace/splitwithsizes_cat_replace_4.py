
class SimpleModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(in_features, out_features, bias=False)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return concatenated_tensor + self.linear1(x1) + self.linear2(x1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            SimpleModule(3, 16),
            torch.nn.ReLU(),
            SimpleModule(16, 20),
            torch.nn.ReLU(),
            SimpleModule(20, 24),
            torch.nn.ReLU()
        )
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
