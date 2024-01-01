
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspo = torch.nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.convTranspo_1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convTranspo_2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convTranspo_3 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convTranspo_4 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convTranspo_5 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2, output_padding=1)
        self.convTranspo_6 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2, output_padding=1)
        self.convTranspo_7 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2, output_padding=1)
        self.convTranspo_8 = torch.nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=2, output_padding=1)
    
    def forward(self, x1):
        h1 = self.convTranspo(x1) # (1 X 64 X 28 X 28)
        h2 = self.convTranspo_1(F.relu(h1))  # (1 X 64 X 28 X 28)
        h3 = self.convTranspo_2(F.relu(h2))  # (1 X 64 X 28 X 28)
        h4 = self.convTranspo_3(F.relu(h3))  # (1 X 64 X 28 X 28)
        h5 = self.convTranspo_4(F.relu(h4))  # (1 X 64 X 28 X 28)
        h6 = self.convTranspo_5(F.relu(h5))  # (1 X 64 X 56 X 56)
        h7 = self.convTranspo_6(F.relu(h6))  # (1 X 64 X 56 X 56)
        h8 = self.convTranspo_7(F.relu(h7))  # (1 X 64 X 56 X 56)
        return F.tanh(self.convTranspo_8(F.relu(h8)))  # (3 X 3 X 112 X 112)
# Inputs to the model
x1 = torch.ones(1, 3, 112, 112)
