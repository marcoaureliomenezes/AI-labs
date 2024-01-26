## Aprendizado Não Supervisionado de Máquina

## Tipos de Aprendizado de Máquina não Supervisionado

- Regras de associação
- Clustering
- Redução de dimensionalidade


## 2 - Regras de associação

### 2.1 - Intrudução

- Regras de associação são usadas para descobrir relações interessantes entre variáveis em grandes bancos de dados.
- São regras do tipo "se-então" que indicam a probabilidade de ocorrência de um evento baseado na ocorrência de outro evento.
- São usadas para identificar relações fortes entre variáveis em grandes bancos de dados transacionais.

### 2.2 - Exemplo

- Suponha que um supermercado queira descobrir relações entre os itens que são frequentemente comprados juntos.
- Então podemos usar regras de associação para descobrir que os clientes que compram pão também compram manteiga com 80% de probabilidade.

### 2.3 - Aplicações

- Cesta de compras
- Agrupamento de documentos: 
- Agrupamento de genes: identificar genes com funções similares
- Agrupamento de notícias: Google usa agrupamento para agrupar notícias similares.
- Agrupamento de imagens: Google usa agrupamento para agrupar imagens similares.
- Agrupamento de vídeos: YouTube usa agrupamento para recomendar vídeos para seus usuários.
- Agrupamento de músicas: Spotify usa agrupamento para recomendar músicas para seus usuários.


### 2.4 - Métricas

- Suporte: probabilidade de um item aparecer na transação.

Formula:

```
Suporte(X) = (Número de transações que contém X) / (Número total de transações)
```
Onde X é um item ou um conjunto de itens.

- Confiança: probabilidade de um item B aparecer na transação dado que o item A apareceu na transação.

Formula:
```
Confiança(X -> Y) = (Número de transações que contém X e Y) / (Número de transações que contém X)
```

## 3 - Clustering

- Agrupamento de dados
- Objetivo: agrupar dados similares