
m = torch.nn.Sequential(
    torch.nn.quantized.FloatFunctional(),
    torch.nn.quantized.DeQuantize()
)

out = m(model(x))
out = clamp_max_(out, min=0.0)
out = clamp_min_(out, min=-128.0)
out = torch.round(out)

m[0].scale = 1./np.array([max_input]*3, dtype=np.float32)
m[0].zero_point = 128

m[1].scale = 1./np.array([127./max_input]*3 + [min_output], dtype=np.float32)
m[1].zero_point = 0

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 2, 3, 4)
