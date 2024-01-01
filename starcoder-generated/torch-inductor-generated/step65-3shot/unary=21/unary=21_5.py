
conv = torch.nn.Conv2d(256, 512, 1, bias=False, stride=1)
linear = torch.nn.Linear(512, 61)
norm = torch.nn.LayerNorm([linear.out_features], elementwise_affine=False, eps=9.99999974738e-06)
tanh = torch.nn.Tanh()
# 50-50 guess is wrong here
model = torch.nn.Sequential(torch.nn.ModuleList([conv, norm, tanh])), torch.nn.ModuleList([linear]))
# Inputs to the model
self = norm
x = torch.randn(64, 256, 8, 8)
