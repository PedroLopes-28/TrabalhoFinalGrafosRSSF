import numpy as np
import random
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import csv
import os

trafego_ch = defaultdict(int)

def salvar_dataset_csv(estatisticas, caminho_csv='espso_base_movel_round_robin1.csv'):
    criar_cabecalho = not os.path.exists(caminho_csv)

    with open(caminho_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'rodada', 'num_CHs_anteriores', 'num_CHs_novos', 'CHs_mantidos',
            'total_mortos', 'bateria_restante'  # <-- alterado aqui
        ])
        if criar_cabecalho:
            writer.writeheader()

        linha = {
            'rodada': estatisticas['rodada'],
            'num_CHs_anteriores': len(estatisticas['CHs_anteriores']),
            'num_CHs_novos': len(estatisticas['CHs_novos']),
            'CHs_mantidos': len(set(estatisticas['CHs_anteriores']).intersection(estatisticas['CHs_novos'])),
            'total_mortos': estatisticas['total_mortos'],  # <-- alterado aqui
            'bateria_restante': estatisticas['bateria_restante']
        }
        writer.writerow(linha)


def extrair_estatisticas_rodada(rodada, CHs_anteriores, CHs_novos, mortos_acumulados, bateria):
    return {
        'rodada': rodada,
        'CHs_anteriores': CHs_anteriores,
        'CHs_novos': CHs_novos,
        'CHs_mantidos': len(set(CHs_anteriores).intersection(CHs_novos)),
        'total_mortos': len(mortos_acumulados),
        'bateria_restante': bateria_total_restante(bateria)
    }

def custo_associacao(no, ch, nos, beta=0.5):
    dist = euclidiana((no.x, no.y), (ch.x, ch.y))
    energia_ch = nos[ch.indice].energia
    return beta * dist + (1 - beta) * (1 / (energia_ch + 1e-6))

@dataclass
class No:
    indice: int
    x: float
    y: float
    energia: float
    distancia_bs: float = 0.0
    tipo: str = ""
    grau: int = 0
    peso: float = 0.0

class MobileSink:
    def __init__(self, posicao_inicial, velocidade=5.0, intervalo_movimento=10):
        self.posicao = np.array(posicao_inicial, dtype=float)
        self.velocidade = velocidade
        self.intervalo_movimento = intervalo_movimento
        self.trajetoria = [posicao_inicial]
        self.proxima_movimento = intervalo_movimento

    def atualizar_posicao(self, nova_posicao):
        self.posicao = np.array(nova_posicao)
        self.trajetoria.append(tuple(nova_posicao))

    def mover_para(self, posicao_alvo, rodada_atual):
        if rodada_atual >= self.proxima_movimento:
            direcao = np.array(posicao_alvo) - self.posicao
            distancia = np.linalg.norm(direcao)
            if distancia > 0:
                passo = min(self.velocidade, distancia)
                self.posicao += (direcao / distancia) * passo
                self.trajetoria.append(tuple(self.posicao))
                self.proxima_movimento = rodada_atual + self.intervalo_movimento
            return True
        return False

class GerenciadorPSO:
    def __init__(self, nos, vizinhos, alpha=0.5, beta=0.5):
        self.nos = nos
        self.vizinhos = vizinhos
        self.alpha = alpha
        self.beta = beta
        self.num_nos = len(nos)
        
        for i, no in enumerate(nos):
            no.grau = len(vizinhos[i])
            no.peso = self.alpha * no.energia + self.beta * no.grau
    
    def calcular_fitness(self, particula, posicao_sink=None):
        energia_total = sum(self.nos[no].energia for no in particula.posicao)
        peso_total = sum(self.nos[no].peso for no in particula.posicao)
        
        penalidade = 0
        for i in range(len(particula.posicao)):
            for j in range(i+1, len(particula.posicao)):
                dist = euclidiana((self.nos[particula.posicao[i]].x, self.nos[particula.posicao[i]].y),
                                 (self.nos[particula.posicao[j]].x, self.nos[particula.posicao[j]].y))
                if dist < 100:
                    penalidade += (100 - dist) * 0.1
        
        fator_sink = 0
        if posicao_sink is not None:
            for ch in particula.posicao:
                dist_sink = euclidiana((self.nos[ch].x, self.nos[ch].y), posicao_sink)
                fator_sink += 1 / (1 + dist_sink)
            for ch in particula.posicao:
                num_membros = len([i for i in range(self.num_nos) if i not in particula.posicao and i in self.vizinhos[ch]])
                penalidade += num_membros * 0.05
                
        return energia_total + peso_total - penalidade + fator_sink * 10

    def atualizar_particula(self, particula, melhor_global, w=0.5, c1=1.5, c2=1.5):
        for i in range(len(particula.posicao)):
            r1 = random.random()
            r2 = random.random()

            vel = (w * particula.velocidade[i] +
                   c1 * r1 * (particula.melhor_posicao[i] - particula.posicao[i]) +
                   c2 * r2 * (melhor_global[i] - particula.posicao[i]))

            particula.velocidade[i] = vel
            nova_pos = int(round(particula.posicao[i] + vel)) % self.num_nos
            particula.posicao[i] = nova_pos

        particula.posicao = list(dict.fromkeys(particula.posicao))[:len(particula.velocidade)]

class Particula:
    def __init__(self, num_nos, num_chs):
        self.posicao = random.sample(range(num_nos), num_chs)
        self.velocidade = [0] * num_chs
        self.melhor_posicao = self.posicao[:]
        self.melhor_fitness = float('-inf')

class ParticulaSink:
    def __init__(self, limites, num_posicoes=5):
        self.limites = limites  # (x_min, x_max, y_min, y_max)
        self.num_posicoes = num_posicoes
        self.posicao = self.inicializar_posicao()
        self.velocidade = self.inicializar_velocidade()
        self.melhor_posicao = self.posicao.copy()
        self.melhor_fitness = float('-inf')
        
    def inicializar_posicao(self):
        return np.array([
            [random.uniform(self.limites[0], self.limites[1]), 
             random.uniform(self.limites[2], self.limites[3])]
            for _ in range(self.num_posicoes)
        ])
        
    def inicializar_velocidade(self):
        return np.random.uniform(-1, 1, (self.num_posicoes, 2))
        
    def atualizar(self, melhor_global, w=0.5, c1=1.5, c2=1.5):
        r1 = random.random()
        r2 = random.random()
        
        for i in range(self.num_posicoes):
            self.velocidade[i] = (
                w * self.velocidade[i] +
                c1 * r1 * (self.melhor_posicao[i] - self.posicao[i]) +
                c2 * r2 * (melhor_global[i] - self.posicao[i])
            )
            
            self.posicao[i] += self.velocidade[i]
            
            self.posicao[i][0] = np.clip(self.posicao[i][0], self.limites[0], self.limites[1])
            self.posicao[i][1] = np.clip(self.posicao[i][1], self.limites[2], self.limites[3])

def eleger_CHs_com_PSO(nos, vizinhos, alpha=0.5, beta=0.5, num_chs=None, num_particulas=20, max_iter=50, posicao_sink=None):
    if num_chs is None:
        num_chs = max(1, int(len(nos) * 0.1))
    
    pso_manager = GerenciadorPSO(nos, vizinhos, alpha, beta)
    enxame = [Particula(len(nos), num_chs) for _ in range(num_particulas)]
    melhor_global = None
    melhor_fitness_global = float('-inf')

    for iteracao in range(max_iter):
        for particula in enxame:
            fitness = pso_manager.calcular_fitness(particula, posicao_sink)

            if fitness > particula.melhor_fitness:
                particula.melhor_fitness = fitness
                particula.melhor_posicao = particula.posicao[:]

            if fitness > melhor_fitness_global:
                melhor_fitness_global = fitness
                melhor_global = particula.posicao[:]

        for particula in enxame:
            pso_manager.atualizar_particula(particula, melhor_global)

    return set(melhor_global)

def otimizar_trajetoria_sink(nos, CHs, limites, num_particulas=10, max_iter=20):
    def calcular_fitness_trajetoria(trajetoria, nos, CHs):
        fitness = 0
        for pos in trajetoria:
            for ch in CHs:
                dist = euclidiana((nos[ch].x, nos[ch].y), pos)
                fitness += 1 / (1 + dist)
                fitness += (trafego_ch.get(ch, 1)) / (1 + dist)
                
            densidade_nos = 0
            for no in nos:
                dist = euclidiana((no.x, no.y), pos)
                if dist < 200:
                    densidade_nos += 1 / (1 + dist)
            fitness += densidade_nos * 0.5
            
        return fitness
    
    enxame = [ParticulaSink(limites) for _ in range(num_particulas)]
    melhor_global = None
    melhor_fitness_global = float('-inf')
    
    for iteracao in range(max_iter):
        for particula in enxame:
            fitness = calcular_fitness_trajetoria(particula.posicao, nos, CHs)
            
            if fitness > particula.melhor_fitness:
                particula.melhor_fitness = fitness
                particula.melhor_posicao = particula.posicao.copy()
                
            if fitness > melhor_fitness_global:
                melhor_fitness_global = fitness
                melhor_global = particula.posicao.copy()
                
        for particula in enxame:
            particula.atualizar(melhor_global)
            
    return melhor_global

def euclidiana(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def inicializar_nos(coords, energia_inicial, sink):
    nos = []
    for i, (x, y) in enumerate(coords):
        dist = euclidiana((x, y), sink.posicao)
        nos.append(No(indice=i, x=x, y=y, energia=energia_inicial, distancia_bs=dist))
    return nos


def construir_vizinhos(nos, distancia_max):
    n = len(nos)
    vizinhos = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidiana((nos[i].x, nos[i].y), (nos[j].x, nos[j].y))
            if dist <= distancia_max:
                vizinhos[i].append(j)
                vizinhos[j].append(i)
    return vizinhos

def limite_membros(tipo, total):
    if tipo == 'pequeno':
        return max(1, math.ceil(0.05 * total))
    elif tipo == 'medio':
        return max(1, math.ceil(0.15 * total))
    else:
        return max(1, math.ceil(0.30 * total))

def formar_clusters(nos, vizinhos, CHs, distancia_max=100):
    total_nos = len(nos)
    clusters = {ch: [] for ch in CHs}
    CHs_set = set(CHs)
    nao_alocados = set(i for i in range(total_nos) if i not in CHs)

    for i in nao_alocados:
        no = nos[i]
        candidatos = [ch for ch in vizinhos[i] if ch in CHs_set]
        candidatos.sort(key=lambda ch: custo_associacao(no, nos[ch], nos))
        if candidatos:
            clusters[candidatos[0]].append(i)
        # Caso contrário, o nó não será alocado (sem CHs vizinhos)

    return clusters

def construir_greedy_cluster(nos_cluster, nos, bs_pos):
    adj = {n: [] for n in nos_cluster}

    for u in nos_cluster:
        min_dist = euclidiana((nos[u].x, nos[u].y), bs_pos)
        melhor_vizinho = None

        for v in nos_cluster:
            if u == v:
                continue
            dist_v = euclidiana((nos[v].x, nos[v].y), bs_pos)
            if dist_v < min_dist:
                melhor_vizinho = v
                min_dist = dist_v

        if melhor_vizinho is not None:
            adj[u].append(melhor_vizinho)
            adj[melhor_vizinho].append(u)

    return adj

def _dfs_contagem_pacotes(u, pai, adj, contagem):
    total_sub = 1
    for v in adj[u]:
        if v == pai:
            continue
        _dfs_contagem_pacotes(v, u, adj, contagem)
        total_sub += contagem[v]
    contagem[u] = total_sub

def _encontrar_pai_em_greedy(u, raiz, adj):
    from collections import deque
    pai = {raiz: None}
    q = deque([raiz])
    visitado = {raiz}

    while q:
        x = q.popleft()
        for v in adj[x]:
            if v not in visitado:
                visitado.add(v)
                pai[v] = x
                q.append(v)
    return pai[u]

def calcular_custo_total(numero_pacotes, distancia_metros):
    V_oper = 3.3
    I_tx = 0.047
    I_rx = 0.010
    tamanho_pacote_bytes = 27

    if distancia_metros <= 500:
        SF = 7
    elif distancia_metros <= 1000:
        SF = 9
    else:
        SF = 12

    Rb_SF = {7: 5470, 9: 1460, 12: 293}
    Rb = Rb_SF.get(SF, 293)
    overhead_bits = 64
    total_bits_pacote = tamanho_pacote_bytes * 8 + overhead_bits
    tempo_no_ar_s = total_bits_pacote / Rb

    P_tx = V_oper * I_tx
    P_rx = V_oper * I_rx

    energia_tx_pacote = P_tx * tempo_no_ar_s
    energia_rx_pacote = P_rx * tempo_no_ar_s

    transmissao = energia_tx_pacote * numero_pacotes
    recepcao = energia_rx_pacote * numero_pacotes

    return transmissao, recepcao

def encaminhar_dados_cluster(nos_cluster, CH, nos, bateria, bs_pos):
    adj = construir_greedy_cluster(nos_cluster, nos, bs_pos)
    contagem = {}
    _dfs_contagem_pacotes(CH, None, adj, contagem)

    mortos = []

    for u in nos_cluster:
        if u == CH:
            continue

        pai = _encontrar_pai_em_greedy(u, CH, adj)

        pacotes_recebidos = 0
        for v in adj[u]:
            if v != pai:
                pacotes_recebidos += contagem[v]

        dist_tx = euclidiana((nos[u].x, nos[u].y), (nos[pai].x, nos[pai].y))
        _, energia_rx_total = calcular_custo_total(pacotes_recebidos, dist_tx)
        energia_tx_total, _ = calcular_custo_total(contagem[u], dist_tx)

        trafego_ch[CH] += contagem[CH]

        bateria[u] -= (energia_rx_total + energia_tx_total)
        if bateria[u] <= 0:
            mortos.append(u)

    pacotes_de_entrada = 0
    for v in adj[CH]:
        pacotes_de_entrada += contagem[v]

    energia_rx_CH = 0.0
    for v in adj[CH]:
        dist_v_ch = euclidiana((nos[v].x, nos[v].y), (nos[CH].x, nos[CH].y))
        _, erx = calcular_custo_total(contagem[v], dist_v_ch)
        energia_rx_CH += erx

    dist_ch_bs = euclidiana((nos[CH].x, nos[CH].y), bs_pos)
    etx_ch, _ = calcular_custo_total(contagem[CH], dist_ch_bs)

    bateria[CH] -= (energia_rx_CH + etx_ch)
    if bateria[CH] <= 0:
        mortos.append(CH)

    return mortos

def remover_nos_mortos(clusters, mortos):
    cluster_com_ch_vazio = {}
    novos = {}

    for ch, membros in clusters.items():
        if ch in mortos:
            sobraram = [m for m in membros if m not in mortos]
            if sobraram:
                cluster_com_ch_vazio[ch] = sobraram
        else:
            membros_sobreviventes = [m for m in membros if m not in mortos]
            novos[ch] = membros_sobreviventes

    return novos, cluster_com_ch_vazio

def reeleicao_CHs_clusters(clusters, nos, viz, alpha=0.5, beta=0.5):
    CHs_anteriores = set(clusters.keys())
    CHs_novos = set()
    novos_clusters = {}

    for ch_antigo, membros in clusters.items():
        candidatos = [ch_antigo] + membros
        candidatos_vivos = [i for i in candidatos if prever_vida_restante(nos[i])]

        melhor = None
        maior_peso = -float('inf')
        for i in candidatos_vivos:
            grau_i = len(viz[i])
            peso_i = alpha * nos[i].energia + beta * grau_i
            if peso_i > maior_peso:
                maior_peso = peso_i
                melhor = i

        if melhor is None:
            continue

        CHs_novos.add(melhor)
        outros = [i for i in candidatos_vivos if i != melhor]
        novos_clusters[melhor] = outros

    return novos_clusters, CHs_anteriores, CHs_novos

def prever_vida_restante(no):
    return no.energia > 0.05

def reeleicao_round_robin(clusters, historico_chs, nos):
    """
    Reeleição de CHs com round-robin considerando apenas nós vivos.
    
    Parâmetros:
    - clusters: dict {CH_antigo: [membros, ...], ...}
    - historico_chs: dict {cluster_id: [lista de CHs usados em ordem]}
    - nos: lista de objetos No para verificar energia
    
    Retorna:
    - novos_clusters: dict {novo_CH: [outros_membros]}
    - CHs_anteriores: set
    - CHs_novos: set
    - novo_historico: dict atualizado
    """
    novos_clusters = {}
    CHs_anteriores = set(clusters.keys())
    CHs_novos = set()
    novo_historico = historico_chs.copy() if historico_chs else {}

    for ch_antigo, membros in clusters.items():
        # Filtra apenas nós com energia positiva
        candidatos = [n for n in ([ch_antigo] + membros) if nos[n].energia > 0]
        
        if not candidatos:
            continue  # Cluster sem nós vivos

        # Obtém histórico ou inicia novo
        usados = novo_historico.get(ch_antigo, [])
        
        # Remove do histórico nós que morreram ou não estão mais no cluster
        usados = [u for u in usados if u in candidatos]
        
        # Encontra próximo nó não utilizado
        restantes = [n for n in candidatos if n not in usados]
        
        if restantes:
            novo_ch = restantes[0]
        else:
            # Reinicia ciclo se todos já foram CHs
            novo_ch = candidatos[0]
            usados = []  # Reseta o histórico para este cluster

        # Atualiza histórico
        novo_historico[ch_antigo] = usados + [novo_ch]
        
        # Cria novo cluster
        outros = [n for n in candidatos if n != novo_ch]
        novos_clusters[novo_ch] = outros
        CHs_novos.add(novo_ch)

    return novos_clusters, CHs_anteriores, CHs_novos, novo_historico

def checar_porcentagem_mudanca(CHs_anteriores, CHs_novos, limiar=1.0):
    if not CHs_anteriores:
        return 1.0, True

    iguais = CHs_anteriores.intersection(CHs_novos)
    total_anteriores = len(CHs_anteriores)
    total_diferentes = total_anteriores - len(iguais)
    frac = total_diferentes / total_anteriores
    return frac, (frac >= limiar)

def rodada_multihop(clusters, nos, viz, bateria, bs_pos, alpha=0.5, beta=0.5, metodo='peso', historico_chs=None):
    todos_mortos_rodada = set()

    # Fase de transmissão de dados
    for ch, membros in list(clusters.items()):
        nos_cluster = [ch] + membros
        mortos_cluster = encaminhar_dados_cluster(nos_cluster, ch, nos, bateria, bs_pos)
        todos_mortos_rodada.update(mortos_cluster)

    # Remove nós mortos dos clusters
    clusters_sem_mortos, cluster_com_ch_vazio = remover_nos_mortos(clusters, todos_mortos_rodada)
    clusters_para_reeleicao = {**clusters_sem_mortos, **cluster_com_ch_vazio}

    # Atualiza energia nos objetos No
    for i in range(len(nos)):
        nos[i].energia = bateria[i]

    # Fase de reeleição de CHs
    if metodo == 'peso':
        novos_clusters, CHs_anteriores, CHs_novos = reeleicao_CHs_clusters(
            clusters_para_reeleicao, nos, viz, alpha, beta
        )
        return novos_clusters, CHs_anteriores, CHs_novos, list(todos_mortos_rodada), historico_chs

    elif metodo == 'round_robin':
        novos_clusters, CHs_anteriores, CHs_novos, novo_historico = reeleicao_round_robin(
            clusters_para_reeleicao, historico_chs or {}, nos
        )
        return novos_clusters, CHs_anteriores, CHs_novos, list(todos_mortos_rodada), novo_historico

    else:
        raise ValueError(f"Método de reeleição inválido: {metodo}")

def coletar_arestas_intracluster(clusters, nos, bs_pos):
    todas_edges = set()
    for ch, membros in clusters.items():
        nos_cluster = [ch] + membros
        adj = construir_greedy_cluster(nos_cluster, nos, bs_pos)
        for u, vizs in adj.items():
            for v in vizs:
                a, b = (u, v) if u < v else (v, u)
                todas_edges.add((a, b))
    return list(todas_edges)

def bateria_total_restante(bateria):
    return sum(e for e in bateria if e > 0)

def conectar_CHs_ate_bs(nos, CHs, bs_pos):
    G = nx.Graph()
    ch_list = list(CHs)

    for i in range(len(ch_list)):
        for j in range(i + 1, len(ch_list)):
            u, v = ch_list[i], ch_list[j]
            dist = euclidiana((nos[u].x, nos[u].y), (nos[v].x, nos[v].y))
            G.add_edge(u, v, weight=dist)

    BS_ID = 'BS'
    for ch in CHs:
        dist = euclidiana((nos[ch].x, nos[ch].y), bs_pos)
        G.add_edge(ch, BS_ID, weight=dist)

    T = nx.minimum_spanning_tree(G)
    interligacoes = [(u, v) for u, v in T.edges() if BS_ID not in (u, v)]
    bs_links = [(u, v) for u, v in T.edges() if BS_ID in (u, v)]

    return interligacoes, bs_links

def visualizar_pso_com_trajetoria(nos, CHs, clusters, edges, sink, rodada=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Cores fixas para cada CH com base no índice
    cores = plt.colormaps['tab20']
    ch_cor = {ch: cores(ch % 20) for ch in CHs}

    # Trajetória do sink
    traj_x, traj_y = zip(*sink.trajetoria)
    ax.plot(traj_x, traj_y, 'y-', alpha=0.3, linewidth=2)
    ax.scatter(traj_x, traj_y, c='yellow', s=30, alpha=0.5)

    # Arestas dos clusters (Greedy intra-cluster)
    for ch, membros in clusters.items():
        cor = ch_cor.get(ch, 'black')
        for u, v in edges:
            if u in membros + [ch] and v in membros + [ch]:
                x1, y1 = nos[u].x, nos[u].y
                x2, y2 = nos[v].x, nos[v].y
                ax.plot([x1, x2], [y1, y2], color=cor, linewidth=2)

    # Interligação entre CHs e CH–BS (linhas tracejadas)
    interligacoes, bs_links = conectar_CHs_ate_bs(nos, CHs, sink.posicao)

    for u, v in interligacoes:
        x1, y1 = nos[u].x, nos[u].y
        x2, y2 = nos[v].x, nos[v].y
        ax.plot([x1, x2], [y1, y2], color='black', linestyle='dotted', linewidth=1.5)

    for u, v in bs_links:
        ch = u if v == 'BS' else v
        x1, y1 = nos[ch].x, nos[ch].y
        x2, y2 = sink.posicao
        ax.plot([x1, x2], [y1, y2], color='black', linestyle='dashed', linewidth=1.5)

    # Plotagem dos nós
    for no in nos:
        x, y = no.x, no.y
        energia = no.energia
        indice = no.indice

        # Verifica se está morto
        morto = energia <= 0
        # Verifica a qual cluster pertence
        dono = next((ch for ch, membros in clusters.items() if indice in membros or indice == ch), None)
        cor = ch_cor.get(dono, 'gray')

        if morto:
            ax.scatter(x, y, c='black', marker='x', s=80, linewidths=2)
        elif indice in CHs:
            ax.scatter(x, y, facecolors='white', edgecolors='black', s=200, marker='o', linewidths=1.5)
        else:
            ax.scatter(x, y, c=[cor], s=100, marker='s')

    ax.scatter(*sink.posicao, facecolors='yellow', edgecolors='black', linewidths=1.5, s=400, marker='*', zorder=10, label='Mobile Sink')

    titulo = 'Mobile Sink PSO'
    if rodada is not None:
        titulo += f' (Rodada {rodada})'
    ax.set_title(titulo)
    ax.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def ler_coords_arquivo(path):
    try:
        with open(path, 'r') as f:
            linhas = [linha.strip() for linha in f if linha.strip()]
            if not linhas:
                raise ValueError("Arquivo está vazio.")
            total = int(linhas[0])
            coords = [tuple(map(float, linha.split(','))) for linha in linhas[1:]]
            if len(coords) != total:
                raise ValueError(f"Número esperado de coordenadas: {total}, mas foram lidas {len(coords)}.")
            print(f"✅ {len(coords)} coordenadas lidas do arquivo.")
            return coords
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo: {e}")
        return []

def calcular_pesos_nos(nos, viz):
    """Calcula os pesos dos nós com base em energia e grau de conectividade"""
    for i, no in enumerate(nos):
        no.grau = len(viz[i])
        no.peso = 0.5 * no.energia + 0.5 * no.grau  # Pesos alpha=0.5 e beta=0.5

def main(metodo_reeleicao='round_robin'):
    # O método de reeleicao pode ser alterado para 'peso'
    path = r'sensores.txt'
    coords = ler_coords_arquivo(path)
    posicao_inicial_sink = (0, 0)
    sink = MobileSink(posicao_inicial_sink, velocidade=100, intervalo_movimento=10)

    energia_inicial = 50.0
    distancia_max = 100
    max_rodadas = 10000
    alpha, beta = 0.5, 0.5
    mortos_acumulados = set()

    all_x = [x for x, y in coords]
    all_y = [y for x, y in coords]
    limites = (min(all_x), max(all_x), min(all_y), max(all_y))

    nos = inicializar_nos(coords, energia_inicial, sink)
    viz = construir_vizinhos(nos, distancia_max)
    calcular_pesos_nos(nos, viz)
    bateria = [energia_inicial for _ in nos]

    # Inicialização do histórico para round_robin
    historico_chs = defaultdict(list) if metodo_reeleicao == 'round_robin' else None

    CHs_iniciais = eleger_CHs_com_PSO(nos, viz, alpha, beta, posicao_sink=sink.posicao)
    clusters = formar_clusters(nos, viz, CHs_iniciais)
    
    membros_sem_CH = [i for i in range(len(nos)) if i not in CHs_iniciais]
    for i in membros_sem_CH:
        if not any(i in membros for membros in clusters.values()):
            clusters[i] = []  # esse nó vira CH sozinho

    CHs_iniciais = set(clusters.keys())  # Atualiza CHs após inclusão

    arestas_iniciais = coletar_arestas_intracluster(clusters, nos, sink.posicao)
    prev_CHs = set(CHs_iniciais)
    visualizar_pso_com_trajetoria(nos, prev_CHs, clusters, arestas_iniciais, sink, rodada=0)

    mortos_plotados = set()

    for rodada in range(1, max_rodadas + 1):
        print(f"\n🔁 Rodada {rodada}")

        # Atualiza informações dos nós
        for no in nos:
            no.distancia_bs = euclidiana((no.x, no.y), sink.posicao)

        # Movimentação do sink
        if rodada % sink.intervalo_movimento == 0:
            posicoes_otimas = otimizar_trajetoria_sink(nos, list(clusters.keys()), limites)
            proximo_alvo = posicoes_otimas[0]
            sink.mover_para(proximo_alvo, rodada)

        # Atualiza pesos dos nós
        for i, no in enumerate(nos):
            no.peso = alpha * no.energia + beta * len(viz[i])

        # Executa a rodada com o método selecionado
        if metodo_reeleicao == 'round_robin':
            clusters, CHs_anteriores, CHs_novos, mortos_rodada, historico_chs = rodada_multihop(
                clusters, nos, viz, bateria, sink.posicao, alpha, beta,
                metodo='round_robin', historico_chs=historico_chs
            )
        else:
            clusters, CHs_anteriores, CHs_novos, mortos_rodada, _ = rodada_multihop(
                clusters, nos, viz, bateria, sink.posicao, alpha, beta, metodo='peso'
            )

        print(f"→ CHs antigos: {sorted(CHs_anteriores)}")
        print(f"→ CHs novos:   {sorted(CHs_novos)}")
        print(f"→ Nós que morreram nesta rodada: {sorted(mortos_rodada)}")
        total_bateria = bateria_total_restante(bateria)
        print(f"🔋 Bateria total restante na rede: {total_bateria:.4f} J")

        mortos_acumulados.update(mortos_rodada)
        estat = extrair_estatisticas_rodada(rodada, CHs_anteriores, CHs_novos, mortos_acumulados, bateria)
        salvar_dataset_csv(estat)

        # Visualização periódica
        novos_mortos = [m for m in mortos_rodada if m not in mortos_plotados]
        if novos_mortos:
            arestas_atual = coletar_arestas_intracluster(clusters, nos, sink.posicao)
            mortos_plotados.update(novos_mortos)
            if novos_mortos:
                print(f"🕵️ Novos mortos visualizados: {novos_mortos}")

        # Verifica se ainda há clusters ativos
        if not clusters:
            visualizar_pso_com_trajetoria(nos, prev_CHs, clusters, arestas_atual, sink, rodada)
            print("✅ Todos os nós morreram ou nenhum cluster restou. Fim da simulação.")
            break

        # Verifica mudança significativa nos CHs
        frac, disparar = checar_porcentagem_mudanca(CHs_anteriores, CHs_novos, limiar=1.0)
        print(f"   * Fração de CHs que mudaram: {frac:.2%}")
        prev_CHs = CHs_novos
        if disparar:
            arestas_atual = coletar_arestas_intracluster(clusters, nos)
            visualizar_pso_com_trajetoria(nos, prev_CHs, clusters, arestas_atual, sink, rodada)

    # Visualização final
    if clusters:
        ultimos_CHs = set(clusters.keys())
        print(f"\n✅ Fim após {rodada} rodadas. CHs finais: {sorted(ultimos_CHs)}")
        arestas_finais = coletar_arestas_intracluster(clusters, nos, sink.posicao)
        visualizar_pso_com_trajetoria(nos, ultimos_CHs, clusters, arestas_finais, sink, rodada)

if __name__ == "__main__":
    main(metodo_reeleicao='round_robin')