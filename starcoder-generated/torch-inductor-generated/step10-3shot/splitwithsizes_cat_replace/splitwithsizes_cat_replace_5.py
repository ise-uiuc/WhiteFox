
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 1, 1), torch.nn.MaxPool2d(3, 1, 1, 0))
        self.split = torch.nn.Conv2d(3, 32, 3, 1, 1)
    def forward(self, x1):
        v1 = self.split(x1)
        split_tensors=self.features(v1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=-1)
        return (concatenated_tensor, torch.split(v1,[1,1,1],dim=-1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
