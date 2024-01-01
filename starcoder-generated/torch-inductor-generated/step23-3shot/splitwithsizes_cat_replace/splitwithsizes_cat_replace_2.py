
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Sigmoid())
        self.pad = torch.nn.Sequential(torch.nn.ConstantPad1d(3, value=0.0))
    def forward(self, x0):
        x4 = self.features(x0)
        x0 = self.pad(x0)
        split_tensors = torch.split(x0, [1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x0, [1, 1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3)
