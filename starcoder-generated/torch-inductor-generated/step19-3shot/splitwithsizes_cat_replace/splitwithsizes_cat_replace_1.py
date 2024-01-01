
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(inplace=True))
        self.split = torch.nn.Sequential(torch.nn.BatchNorm1d(32), torch.nn.Linear(32, 32), torch.nn.ReLU(inplace=True), torch.nn.Linear(32, 1))
        self.concat = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.BatchNorm1d(32), torch.nn.Linear(32, 32), torch.nn.ReLU(inplace=True))
    def forward(self, x1):
        x1 = self.features(x1)
        split_tensors = torch.split(x1, [1, 1, 1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x1, [1, 1, 1, 1, 1], dim=0))
# Inputs to the model
x1 = torch.randn(5, 10, requires_grad=True)
