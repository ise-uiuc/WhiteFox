


# Initializing the model
def f(x, y):
    return x.cat(y, dim=1)[:, 0:9223372036854775807]

# Inputs to the model
x = torch.randn(1, 2, 128, 128)
y = torch.randn(1, 2, 512, 512)
__output_size__ = f(x, y).size(2)
__input__ = [x, y]

