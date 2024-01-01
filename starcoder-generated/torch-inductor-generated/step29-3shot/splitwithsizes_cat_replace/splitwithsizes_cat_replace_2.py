
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(inplace=False)])
        self.classifier = torch.nn.ModuleList([torch.nn.Conv2d(32, 16, 3, 3, 1), torch.nn.Conv2d(16, 8, 3, 3, 1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [4, 4], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        for x in self.features:
            y = x(concatenated_tensor)
        for x in self.classifier:
            y = x(y)
        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
