
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=Fal), torch.nn.ReLU(),)
    def forward(self, x1):
        (v1) = (x1)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
