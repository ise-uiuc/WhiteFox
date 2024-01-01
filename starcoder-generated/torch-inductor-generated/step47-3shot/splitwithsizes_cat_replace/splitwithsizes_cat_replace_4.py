
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 1, 1)])
        self.classifier = torch.nn.ModuleList([torch.nn.Linear(32 * 36 * 36, 2), torch.nn.Linear(2, 2)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        fc = self.classifier[0](concatenated_tensor.view(-1, 36 * 36 * 32))
        fc = self.classifier[1](fc)
        return (fc, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
