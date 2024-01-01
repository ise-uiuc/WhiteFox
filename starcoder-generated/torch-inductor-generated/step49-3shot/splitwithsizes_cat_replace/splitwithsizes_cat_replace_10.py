
model1_1 = torch.nn.ModuleList([Model1()])
model1_2 = torch.nn.ModuleList([Model1()])
model2 = torch.nn.ModuleList([model1_1, model1_2])
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        split_tensors = torch.split(x1, [1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        self.features = torch.nn.ModuleList([model2[0], model2[1], concatenated_tensor])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
