
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList()
        self.num_classes = 10
        self.features.append(torch.nn.Linear(1, self.num_classes, bias = False))
        self.features.append(torch.nn.ReLU())
        self.features.append(torch.nn.Linear(self.num_classes, self.num_classes, bias = False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
