import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Variáveis principais e verificação do da GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Treinando na: {device}")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5

# Coleta as imagens do banco de dados 
transformacao = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transformacao)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transformacao)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Estrutura da rede neural
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# Envia pra GPU e otimizador SGD com LEARNING_RATE = 1e-3 (Versão antiga)
# Nova versão com otimizador Adam e LEARNING_RATE = 1e-2 garantiu melhor resultado com menos épocas.
# 64% de precisão -> 87% de precisão com 5 épocas
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Ensino da rede
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train() 
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_valor, atual = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"Perda (Loss): {loss_valor:>7f}  [{atual:>5d}/{size:>5d}]")

# Teste da rede
def test_loop(dataloader, model, loss_fn):
    model.eval() 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"Resultado do Teste: \n Precisão: {(100 * correct / size):>0.1f}%, Loss: {(test_loss / num_batches):>8f} \n")

# Ciclo de épocas
for t in range(EPOCHS):
    print(f"Época {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Treinamento Concluído!")

# Pega uma imagem do banco de dados e faz a previsão
classes = ["Camiseta", "Calça", "Suéter", "Vestido", "Casaco", 
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]

indice = 4  # Mude este valor para testar outras imagens
imagem, label_real = test_data[indice][0], test_data[indice][1]

model.eval()
with torch.no_grad():
    imagem_gpu = imagem.unsqueeze(0).to(device)
    
    pred = model(imagem_gpu)
    indice_previsto = pred.argmax(1).item()
    
    nome_previsto = classes[indice_previsto]
    nome_real = classes[label_real]

draw_image = imagem.squeeze()

plt.imshow(draw_image, cmap="gray")
cor_do_texto = "green" if indice_previsto == label_real else "red"
plt.title(f"Previsão da rede: {nome_previsto}\nGabarito: {nome_real}", color=cor_do_texto)
plt.axis("off")
plt.show()