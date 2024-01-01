
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat((split_tensors[0], v2, split_tensors[1]), dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 1, 32, 32)
