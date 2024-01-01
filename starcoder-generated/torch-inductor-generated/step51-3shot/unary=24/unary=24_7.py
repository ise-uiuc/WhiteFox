
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = x.shape[3]
        negative_slope = 1.37857933
        v1 = torch.nn.functional.conv2d(x, torch.zeros((32, 1, 1, 1)), bias=torch.ones(32), stride=(1, 4), padding=(0, 0))
        v2 = torch.reshape(1-v1/6, (-1, 32, t))
        v3 = 1+(v1-6)/0.02
        v4 = 0.48681433+(v1-2.73333325)*1.75934
        v5 = 3.9999952+(t-6)/11.424742
        v6 = v2-0.76293294
        v7 = v6/v3*v4+0.7999996
        v8 = (1-torch.abs(v1))
        v9 = v5*torch.round(1+v7)
        v10 = 0.486815+(v8-6)/v9
        v11 = torch.round((1-v10)/torch.max(v10, v1))*1-v10
        v12 = 1+torch.nn.functional.softplus(v11)+(-v11*v10+torch.log(float(-1)))
        v13 = v1*v12
        v14 = torch.where(v13 > 0, v13, v13*negative_slope)
        return v14
# Inputs to the model
x1 = torch.randn(34, 32, 75, 32)
