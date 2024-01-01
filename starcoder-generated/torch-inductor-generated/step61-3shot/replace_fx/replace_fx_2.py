
class Dropout(torch.nn.Module):
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.4, training=self.training, inplace=True)
        return 1
# Inputs to the model
x = torch.randn(1)
