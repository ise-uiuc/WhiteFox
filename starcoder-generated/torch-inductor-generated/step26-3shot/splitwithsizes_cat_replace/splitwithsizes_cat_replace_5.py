
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(1, 3, bias=False))
        self.classifier = torch.nn.Sequential(torch.nn.Sigmoid(), torch.nn.Linear(3, 3, bias=False), torch.nn.Tanh(), torch.nn.Softmax(dim=0))
        self.add_feature = torch.nn.Sigmoid()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, v1.split(1, dim=0))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
