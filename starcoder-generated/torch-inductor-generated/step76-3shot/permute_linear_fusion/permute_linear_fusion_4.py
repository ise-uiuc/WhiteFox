
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.abs(v1)
        v1 = v1 ** 0.9998022947320297
        v1 = torch.sigmoid(v1)
        v1 = v1 ** 2.7585053123405843
        v1 = 0.5 + v1 + v1
        v1 = v1 * 3.710613765903431
        v1 = torch.max(v1, dim=-1)[0]
        v1 = v1 ** 2.469238332158334
        v2 = 0.08808259124998789
        v2 = v2 ** 1.8429539860401728
        v2 = v2 / v2
        v2 = v2 ** 1.3118210001827783
        v2 = v2 + 2.0
        v2 = v2 / 1.3833518390710894
        v2 = 0.028969916391905067 + v2
        v2 = 0.042632822173428655 + v2
        v2 = v2 + 0.455910631267275
        v2 = v2 + 2.745397030699227
        v2 = 0.045973164589283345 + v2
        v2 = v2 + 0.6696742219025932
        v4 = v2.unsqueeze(dim=-1)
        v2 = v2 + v4.to(v2.dtype)
        v1 = v1 + v2
# Inputs to the model
x1 = torch.randn(1, 3, 2)
