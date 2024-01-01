
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = F.dropout(x, p=0.1)
        y2 = F.dropout(x, p=0.1)
        y3 = F.dropout(x, p=0.1)
        y4 = F.dropout(x, p=0.1)
        return output
x=torch.zeros((4, 2)).cuda()
y=Model().cuda()
y(x)
def fn(x:torch.Tensor) -> List[torch.Tensor]:
    x=input_transform(x)
    x=x.cuda()
    x1 = F.dropout(x, p=0.1)
    x2 = F.dropout(x, p=0.1)
    x3 = F.dropout(x, p=0.1)
    x4 = F.dropout(x, p=0.1)
    res = y(x)
    return [res]
input_transform = torch.nn.BatchNorm2d(2)
x=torch.zeros((4, 2)).cuda()
y=Model().cuda()
y(x)


# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        #y1 = F.dropout(x_in, p=0.1, training=False)
        x1 = F.dropout(x, p=0.5, training=False)
        x2 = F.dropout(x, p=0.5, training=True)
        x3 = F.dropout(x1, p=0.5, training=False)
        x4 = F.dropout(x2, p=0.5, training=True)
        return y
x=torch.zeros((4, 2)).cuda()
y=Model().cuda()
y(x)
def fn(x:torch.Tensor) -> List[torch.Tensor]:
    x=input_transform(x)
    x=x.cuda()
    x1 = F.dropout(x, p=0.1, training=False)
    x2 = F.dropout(x, p=0.1, training=False)
    x3 = F.dropout(x, p=0.1, training=False)
    x4 = F.dropout(x, p=0.1, training=False)
    res = y(x)
    return [res]

input_transform = torch.nn.BatchNorm2d(2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = F.dropout(x, p=0.2, training=True)
        y = torch.argmax(x1, dim=1)
        return y
x = torch.zeros((3, 4)).cuda()
y = Model().cuda()
y(x)
def fn(x:torch.Tensor) -> List[torch.Tensor]:
    x=input_transform(x)
    x=x.cuda()
    x1 = F.dropout(x, p=0.1, training=True)
    res = y(x)
    return [res]
input_transform = torch.nn.BatchNorm2d(4)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.dropout(x, p=0.1)
        y1 = torch.rand_like(x)
        y2 = torch.sum(y1)
        y3 = torch.rand_like(x)
        y4 = torch.div(y2,y3)
        return y4
x = torch.zeros((4, 2)).cuda()
y = Model().cuda()
y(x)
def fn(x:torch.Tensor) -> List[torch.Tensor]:
    x=input_transform(x)
    x=x.cuda()
    x1 = F.dropout(x, p=0.1)
    res = y(x)
    return [res]
input_transform = torch.nn.BatchNorm2d(2)


# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = F.dropout(x, p=0.1, training=False)
        x1 = F.dropout(x, p=0.5, training=False)
        x2 = F.dropout(x, p=0.5, training=True)
        x3 = F.dropout(x1, p=0.5, training=False)
        x4 = F.dropout(x2, p=0.5, training=True)
        n1 = torch.norm(x2 - x3)
        n2 = torch.norm(x3 - x4)
        loss = torch.abs(n1 - n2)
        return loss
x = torch.zeros((4, 2)).cuda()
y = Model().cuda()
y(x)
def fn(x:torch.Tensor) -> List[torch.Tensor]:
    x=input_transform(x)
    x=x.cuda()
    x1 = F.dropout(x, p=0.1, training=False)
    x2 = F.dropout(x, p=0.1, training=False)
    x3 = F.dropout(x, p=0.1, training=False)
    x4 = F.dropout(x, p=0.1, training=False)
    n1 = torch.norm(x2 - x3)
    n2 = torch.norm(x3 - x4)
    loss = torch.abs(n1 - n2)
    return [loss]

f=0
g=0
