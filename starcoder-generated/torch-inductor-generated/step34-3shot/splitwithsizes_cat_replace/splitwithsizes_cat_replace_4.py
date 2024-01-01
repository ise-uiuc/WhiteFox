
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Linear(128, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10)])
        self.fc1 = torch.nn.Linear(10, 16)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        output1 = torch.add(self.features[0](concatenated_tensor), self.features[1](concatenated_tensor))
        output2 = torch.add(self.features[2](output1), self.features[3](output1))
        return (concatenated_tensor, self.features(v1), (self.features(v1), output2), v1, split_tensors, concatenated_tensor, self.fc1(output2))
# Inputs to the model
x1 = torch.randn(1, 10, 1)
