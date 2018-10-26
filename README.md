# aq30m

** Todos casos de testes tem:
*** um modelo de 70% e teste de 30%

teste1 : 
* Utiliza o conhecimento_filtrado:
* Melhores colunas para o modelo

* Resultado:
* Acertos: 19734/21497 (91%)

teste2 : 
* Utiliza o conhecimento_filtrado:
* Todas colunas disponíveis

* Resultado:
* Acertos: 20651/21497 (96%)


teste3 : 
* Utiliza o conhecimento_filtrado, quantidade de verifica 1 e 3 iguais:
* Melhores colunas para o modelo

* Resultado:
* Acertos: 20336/21497 (94%)


teste4 : 
* Utiliza o conhecimento_filtrado, quantidade de verifica 1 e 3 iguais:
* Todas colunas disponíveis

* Resultado:
* Acertos: 21104/21497 (98%)


- Gerar uma amostra de 1000
- Pegar esta amostra e remove-la do conhecimento
- Limpar tag verifica
- Bateria de teste com conhecimentos diferentes(todos conhecimentos sem a amostra atual avaliada):
-- Conhecimento completo e colunas selecionadas - ok
-- Conhecimento completo e TODAS colunas - ok
-- Conhecimento gerado com AAS(amostra aleatoria simples) e colunas selecionadas - ok
-- Conhecimento gerado com AAS(amostra aleatoria simples) e TODAS colunas - ok
-- Conhecimento trimestre historico
- Gerar matriz de confusão para cada teste