
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ModuleList([torch.nn.ELU(alpha=1.7597629366066112) for i in range(1)], 0, 2), torch.nn.Hardsigmoid(), torch.nn.Hardtanh(), torch.nn.Hardswish(), torch.nn.PReLU(), torch.nn.RReLU(0.7181348807948663, 0.273423160986779))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
