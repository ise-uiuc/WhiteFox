
class ModelTanh(torch.nn.Module):
    def forward(self, x):
        v1 = torch.tanh(x)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
