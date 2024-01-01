
def generate_model(device):
    model = torch.nn.Transformer(d_model=8, num_encoder_layers=1, num_decoder_layers=1)
    model = model.train()
    model.to(device)
    return model

x1 = torch.randn(4, 6, 8)
x2 = torch.randn(4, 4, 8)
model = generate_model('cpu')
