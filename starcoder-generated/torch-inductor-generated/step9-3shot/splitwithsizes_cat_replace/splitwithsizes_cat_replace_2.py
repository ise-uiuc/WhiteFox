
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
        self.split = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.Tanh())
    def forward(self, x1):
        x2 = self.features(x1)
        split_tensors = torch.split(x2, [2, 1, 2], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, x2, torch.split(x2, [2, 1, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 5, 12, 12)
