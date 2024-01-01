
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.rand(2, 3, 3, 3)
        self.features2 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.features3 = torch.nn.Conv2d(64, 3, 3, 1, 1, bias=False)
    def forward(self, x, y):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        split_tensors1 = torch.split(x1, 2, dim=1)
        concatenated_tensor = torch.cat(split_tensors1, dim=1)
        return (concatenated_tensor, x3)
# Inputs to the model
torch.manual_seed(0)
x1 = torch.randn(1, 64, 1, 1)
y1 = torch.randn(1, 3, 1, 1)
