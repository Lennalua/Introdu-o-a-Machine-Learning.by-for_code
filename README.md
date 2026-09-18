# Introducao-a-Machine-Learning.by-for_code
O arquivo em python anexado trata da atividade final do curso de machine learning ministrado pela liga de programação de for_code da UFRJ. O projeto incluiu a exploração dos dados, treino dos modelos, ajuste de hiperparâmetros (profundidade) para evitar overfitting e comparação de resultados. Concluiu-se com a extração da importância das features, garantindo a melhor aplicabilidade e interpretação do modelo para este conjunto de dados

Questões:
1. Qual é a diferença entre uma árvore isolada e uma floresta de árvores? 
Uma árvore isolada (Decision Tree) um modelo que cria um conjunto de regras de decisão, ramos e folhas. Uma floresta, Random Forest, é um conjunto de inúmeras árvores de decisão, a floresta consulta todas as suas árvores e toma a decisão final ponderando as respostas
2. O que pode acontecer se max_depth for muito alto? Se o parâmetro max_depth for alto, o modelo irá sofrer de overfitting, criando regras específicas até para exceções. Como resultado, a acurácia no treino será próxima de 100%, mas o modelo não será adequado 
3. O que muda ao aumentar n_estimators? O parâmetro n_estimators define a quantidade de árvores que vão compor o modelo Random Forest, melhorando a estabilidade e o desempenho geral do modelo, tornando-o menos propenso ao overfitting. No entanto, a partir de um certo ponto de saturação, adicionar mais árvores não melhora significativamente a acurácia, servindo apenas para tornar o treino do modelo mais lento e pesado 
4. Qual feature foi considerada mais importante? Intensidade da cor 
5. A importância da feature significa que ela causa o resultado? Justifique. Não. A importância indica apenas uma forte correlação estatística e o quanto essa feature foi matematicamente útil para o modelo, não implicando causalidade 

