
class ModelTanh(torch.nn.Module):
    def forward(self, x):
        t1 = x * x
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(256, 256)
