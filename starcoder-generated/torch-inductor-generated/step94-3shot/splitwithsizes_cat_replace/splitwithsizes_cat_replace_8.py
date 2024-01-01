
def block1():
    return torch.nn.Sequential(*(torch.nn.Conv2d(8, 32, 3, 1, 1), torch.nn.Conv2d(32, 8, 3, 1, 1)))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.Sequential(*[block1() for _ in range(4)])
    def forward(self, X):
        split_tensors = torch.split(X, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(X, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
