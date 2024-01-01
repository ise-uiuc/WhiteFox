
class Model(torch.nn.Module):
    def forward(self, x1):
        x2 = torch.zeros_like(x1, dtype=torch.float, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
        x3 = torch.rand_like(x1, dtype=torch.float, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
# Inputs to the model
        x4 = torch.rand_like(x1, dtype=torch.float, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
