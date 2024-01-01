
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.interpolate(x2, size=5, scale_factor=0.125, mode='bilinear', align_corners=False)
        v4 = torch.argmax(v3, dim=1).unsqueeze(dim=1)
        v4 = torch.nn.functional.embedding(v4, self.linear.weight)
        v5 = (v3 == v2).to(v3.dtype)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
