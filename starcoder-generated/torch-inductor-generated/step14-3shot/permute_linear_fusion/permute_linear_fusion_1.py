
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.layer_norm = torch.nn.LayerNorm(shape=(2, 2))
    def forward(self, x1):
        v1 = self.layer_norm(x1)
        v2 = x1.permute(0, 2, 1)
        v3 = torch.tensor([[[0., 1.]], [[1., 0.]]], requires_grad=True)
        v3 = v3.to(device=v1.device, dtype=v1.dtype)
        self.linear.weight = v3
        return self.linear(v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
