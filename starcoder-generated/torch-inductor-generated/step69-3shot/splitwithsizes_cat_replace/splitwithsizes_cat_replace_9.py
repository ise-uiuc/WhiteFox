
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Linear(1, 1)]
        self.features = torch.nn.Sequential(*block_0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1)
# Inputs to the model
x1 = torch.randn(3, 1)
