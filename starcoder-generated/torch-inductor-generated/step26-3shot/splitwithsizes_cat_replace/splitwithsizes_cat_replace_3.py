
# torch.split with the only argument specified as an integer
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if False:
            self.maxpooling2d = torch.nn.Sequential(torch.nn.MaxPool1d(11, 1))
        self.flatten = torch.nn.Sequential(torch.nn.Flatten())
    def forward(self, v1):
        split_tensors = torch.split(v1, 2)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, 2))
# Inputs to the model
in_features = 20
x1 = torch.randn(1, in_features)
