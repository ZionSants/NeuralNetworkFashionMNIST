# NeuralNetworkFashionMNIST

Rede neural do tipo Multilayer Perceptron desenvolvida em PyTorch. O modelo treina para reconhecer e classificar 10 categorias diferentes de peças de roupa utilizando o dataset **FashionMNIST**.

- Suporte automático para aceleração via GPU (CUDA) se o hardware estiver disponível.
- Utiliza `matplotlib` para exibir imagens do conjunto de testes junto com a previsão da rede e o gabarito real em uma interface gráfica.

## Tecnologias Utilizadas

- **Python 3.14.7**
- **PyTorch** 
- **Torchvision** 
- **Matplotlib** 

## Melhorias Futuras

- Implementar o salvamento dos pesos do modelo, permitindo carregar a IA já treinada instantaneamente para testes futuros.
- Substituir as camadas lineares por camadas convolucionais, transformando o modelo em uma CNN para aumentar a precisão na identificação das peças de roupa.
