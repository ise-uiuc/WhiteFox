
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv_t = nn.ConvTranspose2d(1,1,3)
    def forward(self, x):
        x = self.conv_t(x)
        return x[:, :, 1:4:2, 2:4]
# Inputs to the model
input_rand = torch.rand(1, 1, 3, 3).numpy()
x1 = []
x1.append(input_rand)
x1 = torch.tensor(x1, requires_grad=True)

result_output = x1.cpu().detach().numpy()
output_shape = result_output[0].shape

# Outputs to the model
output = []
for f in result_output:
    l = []
    for d in f.flatten().tolist():
        l.append(d)
    output.extend(l)
