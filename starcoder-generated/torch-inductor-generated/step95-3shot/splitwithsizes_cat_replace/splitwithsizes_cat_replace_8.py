
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.Sequential(*(torch.nn.Conv2d(3, 4, 3, 1, 1) for _ in range(3)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 2], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 2], dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
