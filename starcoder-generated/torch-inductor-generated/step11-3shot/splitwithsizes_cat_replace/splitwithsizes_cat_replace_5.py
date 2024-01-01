
class Split_with_Sizes_Concat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3, 1, 0), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2, 0))
    def forward(self, x1):
        v1 = self.features(x1)
        return (v1,)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
