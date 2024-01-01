
class foo(torch.nn.Module):
    def forward(self, x):
        # Dropout module will be removed due to the pattern
        x = torch.nn.Dropout(p=0.5)(x)
        a = torch.rand(1)
        return x
# Inputs to the model
x = torch.rand(1)
