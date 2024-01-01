
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # [torch.Tensor(8,224,224)]
        self.fc1 = torch.nn.Linear(224, 224)
        # [torch.Tensor(8,224,224)]
        self.fc2 = torch.nn.Linear(224, 224)
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.fc2(v1)
        return v2
# Inputs to the model
x = torch.randn(8, 224, 224)
