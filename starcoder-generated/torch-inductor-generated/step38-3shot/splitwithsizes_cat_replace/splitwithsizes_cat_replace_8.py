
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.PReLU()
    def forward(self, v1):
        v3 = torch.randn(1, 5)
        split_tensors = torch.split(v3, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v3, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
