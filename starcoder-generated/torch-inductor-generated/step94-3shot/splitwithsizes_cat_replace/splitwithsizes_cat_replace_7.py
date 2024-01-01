
class Model(torch.nn.Module):
    def forward(self, v1):
        a = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors = (a[0], a[1], a[2])
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
