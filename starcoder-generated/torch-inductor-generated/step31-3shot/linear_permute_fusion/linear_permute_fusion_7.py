
w = 3
x = torch.randn(w, 2, w)
y = torch.nn.functional.linear(x, torch.zeros([w, w], device='cpu'), bias=None)
# Inputs to the model
x1 = torch.randn(w, w, w)
y = torch.nn.functional.linear(x, x1, bias=None)
