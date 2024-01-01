
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Hardtanh(inplace=False)
    def forward(self, v1):
        split_tensors1 = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors2 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors1[i] + split_tensors2[i] for i in range(len(split_tensors1))], dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
