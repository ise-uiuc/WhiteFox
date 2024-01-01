
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.ModuleList([torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)])
        self.classifierlist = torch.nn.ModuleList([copy.deepcopy(self.classifier[0])])
    def forward(self, x1):
        split_tensors = torch.split(self.classifier, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors = torch.split(self.classifierlist, [1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, torch.split(x1, [1, 1, 1], dim=0), split_tensors, torch.split(self.classifier, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
