
def model(x):
    y = torch.cat(torch.stack((torch.stack((x, x)), torch.stack((x, x)))), dim=1)
    return y.view(-1, 4)
# Inputs to the model
x = torch.randn(2, 1, 4)
