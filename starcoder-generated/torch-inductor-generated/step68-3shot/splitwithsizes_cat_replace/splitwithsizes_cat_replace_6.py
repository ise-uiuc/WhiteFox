
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ReLU(inplace=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1], dim=0) # torch.split(v1, [1, 1, 0], dim=0) is not the desired case for this pattern
        split_tensors[0] = torch.squeeze(split_tensors[0], dim=0)
        split_tensors[1] = torch.squeeze(split_tensors[1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1], dim=0))
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
