
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convx = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.split_tensors = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 3, 1, 1, bias=False), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False))
    def forward(self, v1):
        concatenated_tensor = torch.cat([v1, v1], dim=1)
        split_tensors = torch.split(concatenated_tensor, [1, 1, 1], dim=1)
        merged_tensor = self.split_tensors(torch.cat(split_tensors, dim=1))
        merged_tensor = torch.cat([merged_tensor, merged_tensor], dim=1)
        return (merged_tensor, torch.split(v1, [1, 32], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
