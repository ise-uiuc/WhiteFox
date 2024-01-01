
class ModelTanh(torch.nn.Module):
    def forward(self, x2):
        v2 = torch.tanh(x2)
        return v2
# Inputs to the model
x2 = torch.randn(10, 128, 16, 16)
