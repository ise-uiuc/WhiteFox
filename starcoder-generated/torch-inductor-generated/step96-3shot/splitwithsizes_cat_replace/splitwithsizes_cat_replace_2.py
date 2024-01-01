
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.MaxPool2d(5, 1, 2)
        self.features3 = torch.nn.MaxPool2d(1, 1, 4)
        features2 = torch.split(torch.split(torch.split(v1, [1, 1, 1], dim=1), [1, 1, 1], dim=1), [1, 1, 1], dim=1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(v1), v1), v1), v1), v1), v1), v1), v1), v1), v1))
# Inputs to the model
x1 = torch.randn(1, 10, 2)
