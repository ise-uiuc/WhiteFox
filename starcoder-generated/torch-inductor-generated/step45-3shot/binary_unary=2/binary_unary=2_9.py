
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = TinyModel()
        self.model2 = TinyModel()
        self.model3 = TinyModel()
    def forward(self, x1):
        v1 = self.model1(x1)
        v2 = self.model2(v1)
        v3 = v2 - 5
        v4 = F.relu(v3)
        v5 = self.model3(v4)
        v6 = v5 - 4
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
