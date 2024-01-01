
class Model(torch.nn.Module):
    def forward(self, x):
      input228 = torch.split_with_sizes(x, [1, 1, 225, 225], 1)
      v0 = []
      v1 = 0
      for i in range(len(input228)):
        v1 = v1 + 1
        v2 = input228[v1].squeeze(1)
        v3 = torch.unsqueeze(v2, 0)
        v4 = [v0.append(v3) for _ in range(v2.size(0))]
      v5 = torch.cat(v0, 0)
      return v5

# Initializing the model and generating the input tensor
m = Model()
x = torch.randn(25, 6)

