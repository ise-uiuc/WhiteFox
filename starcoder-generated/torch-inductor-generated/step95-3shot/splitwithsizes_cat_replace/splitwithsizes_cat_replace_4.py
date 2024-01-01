
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.eye(256)
        self.features_1 = torch.nn.Linear(256, 128, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v0 = self.features_1(self.features.mm(torch.squeeze(split_tensors[1], dim=1).t()))
        v2 = torch.cat([self.features_1(self.features_1(self.features.mm(
            torch.squeeze(split_tensors[2], dim=1).t())))], dim=1)
        return (concatenated_tensor, v0, v2, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3)
