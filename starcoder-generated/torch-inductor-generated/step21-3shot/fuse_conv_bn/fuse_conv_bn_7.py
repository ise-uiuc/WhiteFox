
def gen_model_conv_bn_relu_conv():
    m = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(), torch.nn.Conv2d(4, 4, 2))
    weights = [torch.rand_like(p) for p in m.parameters()]
    torch.manual_seed(2)
    for p, w in zip(m.parameters(), weights):
        p.data = w
    return m
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
