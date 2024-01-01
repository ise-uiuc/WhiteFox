
class ModelWithSubtraction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(224, 24)
        self.linear2 = torch.nn.Linear(224, 7)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.randn(10, 224)
        v3 = self.linear2(v2)
        v4 = v1 + v3
        v5 = F.relu(v4)
        return v5

# Initializing the model
m = ModelWithSubtraction()

# Inputs to the model
__input_data__ = torch.randn(10, 224)
