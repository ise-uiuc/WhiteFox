
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        # pattern_match_0
        v0 = torch.cat((a, a), dim=0)
        v1 = v0.view((2, a.size(0), a.size(1))).permute((1, 0, 2)).contiguous().clone()
        # pattern_match_0_2
        v3 = v0.view((a.size(0), 2 * a.size(0), a.size(1))).permute((1, 0, 2)).contiguous().clone()

        v2 = torch.relu(v1)
        v4 = torch.relu(v3)
        return v2 * v4.view(-1)
# Input for the model
a = torch.randn(2, 3)
