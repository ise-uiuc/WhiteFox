
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(inplace=False))
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=1)
        split_tensors_v2 = torch.split(v2, [3, 3], dim=1)
        concatenated_tensor = torch.cat([concatenated_tensor, split_tensors_v2[i] for i in range(len(split_tensors_v2))], dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1), torch.split(v2, [3, 3], dim=1))
# Inputs to the model
x1 = torch.randn(3, 1, 2, 2)
x2 = torch.randn(1, 3, 64, 64)
