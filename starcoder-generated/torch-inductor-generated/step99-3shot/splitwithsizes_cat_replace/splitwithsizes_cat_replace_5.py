
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ReLU(inplace=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        tmp = torch.stack(split_tensors, dim=0)
        concatenated_tensor = torch.cat(tmp, dim=0)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
