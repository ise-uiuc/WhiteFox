
class ModelTanh(torch.nn.Module):
    def forward(self, x):
        v1 = F.relu(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(8, 8, 8 )
