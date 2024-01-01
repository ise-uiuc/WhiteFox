
def model(x1, x2):
    x3 = torch.nn.functional.dropout(x1, p=0.2)
    x4 = torch.nn.functional.dropout(x2, p=0.2, training=False)
    return x3 + x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
