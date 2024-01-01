
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block * 3)
        self.classifier = torch.nn.Linear(2592, 2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [32, 32, 32], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        output = self.features(concatenated_tensor)
        output = F.max_pool2d(output, 4)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return (output)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
