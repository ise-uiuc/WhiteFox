
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        v1 = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim=1)
        