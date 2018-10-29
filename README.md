# aq30m

> Nestes testes, foi separado um modelo de 70% do conhecimento e o resto de 30% para ser classificado

## Bateria de teste para classificação em massa:
- Teste1 - Conhecimento completo e colunas selecionadas
    - Acertos: 94644/116565 (81%)
- Teste2 - Conhecimento completo e TODAS colunas
    - Acertos: 102985/116565 (88%)
- Teste3 - Conhecimento gerado com AAS(amostra aleatoria simples) e colunas selecionadas
    -  Acertos: 100422/116565 (86%)
- Teste4 - Conhecimento gerado com AAS(amostra aleatoria simples) e TODAS colunas
    - Acertos: 107998/116565 (92%)

## Bateria de teste para classificação de uma passagem aleatória:
- Teste5 - Conhecimento completo e colunas selecionadas
    - Acertos: 320/930 (34%)
LS82210672017205
- Teste6 - Conhecimento completo e TODAS colunas
    - Acertos: 3099/3203 (96%)
LS82210672016347
- Teste7 - Conhecimento gerado com AAS(amostra aleatoria simples) e colunas selecionadas
    - Acertos: 3009/4610 (65%)
LS82210742017317
- Teste8 - Conhecimento gerado com AAS(amostra aleatoria simples) e TODAS colunas
    - Acertos: 6946/7645 (90%)
LS82210742016283
- Teste9 - Conhecimento do trimestre historico com todas orb_pto
    - Acertos: 778/1410 (55%)
LS82210672016123
- Teste10 - Conhecimento trimestre historico da mesma orb_pto
    - Acertos: 3287/4444 (73%)
    LS82210742016315

Obs:
> Gerar matriz de confusão para cada teste

## TODO
- Gerar uma amostra de 1000
- Pegar esta amostra e remove-la do conhecimento
- Limpar tag verifica
