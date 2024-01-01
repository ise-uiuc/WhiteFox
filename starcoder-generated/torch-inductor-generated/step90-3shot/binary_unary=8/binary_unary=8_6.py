
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.nn.functional.layer_norm(x1, [32, 3, 5, 5])
        v2 = torch.nn.functional.layer_norm(v1, [32, 3, 5, 5])
        v3 = v2 + x1
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 224, 224)
