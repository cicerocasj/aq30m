import pandas as pd
import numpy as np
def get_melhores_colunas(df_rank):
    label_grupo_bom = df_rank[df_rank[1]==df_rank[1].max()].labels.values[0]
    melhores_colunas = df_rank[df_rank['labels']==label_grupo_bom][0]
    cols = melhores_colunas.values.tolist()
    return cols
def get_rank(arquivo):
    df_conhecimento = pd.read_csv(arquivo)
    # Criando classificador RF
    rf_clas = RandomForestClassifier(n_jobs=8, n_estimators=100, max_features=5, random_state=0, max_depth=3, min_samples_split=5)
    rf_clas.fit(df_conhecimento[colunas_stats], df_conhecimento.verifica)
    
    # Criando rank colunas
    df_rank = pd.DataFrame(sorted(list(zip(colunas_stats,rf_clas.feature_importances_)), key=lambda x: x[1], reverse=True))
    
    # Divide colunas em 2 grupos
    dados = df_rank[[1]].values
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, n_jobs=8)
    kmeans = kmeans.fit(dados)
    labels = kmeans.predict(dados)
    df_rank['labels'] = labels
    
    return df_rank

def normaliza(dado):
    mi = dado.min()
    ma = dado.max()
    d = ((dado-mi) / (ma-mi))
    return d
def gera_sumario(df):
    dados = []
    for c in cols:
        d = normaliza(df[c])
        
        dados.append([
            c, # coluna
            d.mean(),   # media
            d.median(), # mediana
            d.std()**2, # variancia
            d.std(),    # std
        ])
    sumario = pd.DataFrame(dados, columns=['coluna', 'media', 'mediana', 'variancia', 'std'])
    sumario = sumario.sort_values('std', ascending=False).copy().reset_index(drop=True)
    return sumario

def configura_df(arquivo):
    global df
    global df1
    global df3
    global df3_full
    df = pd.read_csv(arquivo)
    
    df1 = df[df.verifica==1].copy().reset_index(drop=True)
    df3_full = df[df.verifica==3].copy().reset_index(drop=True)
    
    ids_3 = np.random.choice(df3_full.id, len(df1), replace=False)
    df3 = df[df.id.isin(ids_3)].copy().reset_index(drop=True)
    
    print(arquivo)
    print('df', len(df))
    print('df1', len(df1))
    print('df3', len(df3))
    print('df3_full', len(df3_full))