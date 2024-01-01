
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.rand_like(x1)
        x2 = torch.nn.functional.dropout(x1, p=0.5, training=True)
        x3 = torch.rand_like(x2)
        x4 = F.dropout(x2, p=0.5)
        x5 = torch.nn.functional.dropout(x2, p=0.3, training=True)
        x6 = torch.rand_like(x5)
        x7 = F.dropout(x5, p=0.5)
        x8 = torch.rand_like(x7)
        x9 = F.dropout(x7, p=0.5)
        x10 = torch.rand_like(x9)
        x11 = F.dropout(x9, p=0.5)
        x12 = torch.rand_like(x11)
        x13 = torch.nn.functional.dropout(x11, p=0.300000001192092896)
        return x13
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.rand_like(x1)
        x2 = torch.nn.functional.dropout(x1, p=0.5, training=True)
        x3 = torch.rand_like(x2)
        x4 = F.dropout(x2, p=0.5)
        x5 = torch.nn.functional.dropout(x2, p=0.3, training=True)
        x6 = torch.rand_like(x5)
        x7 = F.dropout(x5, p=0.5)
        x8 = torch.rand_like(x7)
        x9 = F.dropout(x7, p=0.5)
        x10 = torch.rand_like(x9)
        x11 = F.dropout(x9, p=0.5)
        x12 = torch.rand_like(x11)
        x13 = torch.nn.functional.dropout(x11, p=0.300000001192092896)
        return x12
# Inputs to the model
x1 = torch.zeros(1, 2, 2)
