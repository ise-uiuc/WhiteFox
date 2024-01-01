
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        block = [torch.nn.Conv2d(3, 32, 3, 1 - 1 + 1, 0 - 0 + 1, bias=False)]
        self.features = torch.nn.Sequential(*block * 2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)

        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
