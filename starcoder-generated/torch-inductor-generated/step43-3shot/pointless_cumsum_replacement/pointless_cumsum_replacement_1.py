
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([976, 1024], -0.00051707, dtype=torch.float16, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t2 = torch.exp(t1)
        t3 = t2 * torch.sqrt(torch.tensor(795.94091796875)).to(dtype=torch.float16)
        t4 = torch.rand(976, 1024, dtype=torch.float16, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t5 = t4 * t3
        t6 = t5 * -0.03736045416835785
        t7 = t6 + 0.013442041213035583
        t8 = torch.sigmoid(t7)
        t9 = t8 * 0.05528832020091057
        t10 = t9 + 0.003260288772518639
        t11 = t10 - 0.013401031310820584
        t12 = t11 * -0.008240049831323624
        t13 = t12 + 0.018585406033706665
        t14 = torch.ceil(t13)
        t15 = t14 - 1.0
        t16 = torch.mul(t15, -0.010231505373716354)
        t17 = t16 + 0.02776596259784794
        t18 = torch.mul(t17, -0.01909807580719948)
        t19 = t18 - 0.009946531035437584
        t20 = t19 - 0.8235650873184204
        t21 = torch.mul(t20, 2.0647900581359863)
        t22 = t21 + 1.5
        return t22
# Inputs to the model
x1 = torch.randn(976, 1024, device='cuda:0')
