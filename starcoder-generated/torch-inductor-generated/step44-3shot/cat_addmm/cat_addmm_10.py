
def model_factory_func_1(in_features, out_features, use_bias):
    layer = nn.Linear(in_features, out_features, use_bias)
    return nn.Sequential(OrderedDict([("0", layer),
                                        ("1", nn.Tanh())]))
# Inputs to the model
x = torch.randn((2), requires_grad=True)
in_features = 3
out_features = 2
use_bias = True
