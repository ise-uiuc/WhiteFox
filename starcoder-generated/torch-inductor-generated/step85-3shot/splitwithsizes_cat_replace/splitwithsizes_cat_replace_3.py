
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.BatchNorm1d(num_features=32, eps=9.999999747378752e-06, momentum=0.0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
