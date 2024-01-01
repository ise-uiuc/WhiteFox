
y1 = x1*torch.rnormal(torch.full_like(x1, 100.0), torch.full_like(x1,.01))*x1*torch.rnormal(torch.full_like(x1, 111.0), torch.full_like(x1,.01))*x1*torch.rnormal(torch.full_like(x1, 97.0), torch.full_like(x1,.01))*x1
# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)
