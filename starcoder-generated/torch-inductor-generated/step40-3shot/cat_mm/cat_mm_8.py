
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = []
        shape1 = x1.size()
        shape2 = x2.size()
        shape3 = x3.size()
        v.append(torch.mm(x1, x2))
        v.append(torch.mm(x1, x2))
        if (len(shape1) == 2) or (len(shape2) == 2) or (len(shape3) == 2):
            v.append(torch.nn.functional.interpolate(x1, (100, 100), mode='bilinear', align_corners=False))
            v.append(torch.nn.functional.interpolate(x1, (105, 105), mode='bilinear', align_corners=False))
        i = 0
        for x in v:
            if i < 4:
                v[i] = x + torch.mm(v[i], v[i])
            i = i+1
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
