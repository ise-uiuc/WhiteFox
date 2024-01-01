
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.flatten(start_dim=1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 512, 14, 14)
