
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False),torch.nn.ReLU(),torch.nn.Conv2d(32, 32, 1, 1, 1, bias=False),torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False),torch.nn.ReLU()])
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
