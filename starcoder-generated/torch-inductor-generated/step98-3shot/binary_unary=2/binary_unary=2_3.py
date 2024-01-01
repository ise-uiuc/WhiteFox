
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.size(1)
        v2 = x1.size(2)
        v3 = x1.permute([0, 2, 3, 1]) - 3
        v4 = F.relu(v3)
        v5 = v4.permute([0, 3, 1, 2])
        v6 = v5.mul(3)
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.rand((1, 3, 12, 15))
