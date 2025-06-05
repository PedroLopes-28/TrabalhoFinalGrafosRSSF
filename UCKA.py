import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx



@dataclass
class No:
    indice: int
    x: float
    y: float
    energia: float
    distancia_bs: float = 0.0
    omega: float = 0.0
    tipo: str = ""

def euclideana(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def inicializar_nos(coords, energia_inicial, bs_pos):
    dists = [euclideana((x, y), bs_pos) for x, y in coords]
    dmin, dmax = min(dists), max(dists)
    nos = []
    for i, (x, y) in enumerate(coords):
        dist = euclideana((x, y), bs_pos)
        omega = calcular_omega(dist, dmin, dmax)
        tipo = classificar_omega(omega)
        nos.append(No(indice=i, x=x, y=y, energia=energia_inicial,
              distancia_bs=dist, omega=omega, tipo=tipo))

    return nos


def calcular_omega(dist, dmin, dmax):
    return (dist - dmin) / (dmax - dmin)

def classificar_omega(omega):
    if omega <= 1/3:
        return "small"
    elif omega <= 2/3:
        return "medium"
    else:
        return "large"


def classificar_nos(coords, energia_inicial, bs_pos):
    distancias = [euclideana((x, y), bs_pos) for x, y in coords]
    dmin, dmax = min(distancias), max(distancias)

    nos = []
    for i, (x, y) in enumerate(coords):
        dist = distancias[i]
        omega = (dist - dmin) / (dmax - dmin)
        if omega <= 1/3:
            tipo = "small"
        elif omega <= 2/3:
            tipo = "medium"
        else:
            tipo = "large"
        nos.append(No(i, x, y, energia_inicial, dist, omega, tipo))
    return nos

def construir_vizinhos(nos, max_dist=250):
    n = len(nos)
    vizinhos = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclideana((nos[i].x, nos[i].y), (nos[j].x, nos[j].y))
            if dist <= max_dist:
                vizinhos[i].append(j)
                vizinhos[j].append(i)
    return vizinhos
def calcular_peso(energia, grau, alpha=0.5, beta=0.5):
    return alpha * energia + beta * grau

def eleger_CHs(nos, vizinhos, alpha=0.5, beta=0.5):
    # Inicializa o atributo peso para todos os nós
    for no in nos:
        no.peso = 0

    CHs = set()
    for i, no in enumerate(nos):
        grau = len(vizinhos[i])
        peso = calcular_peso(no.energia, grau, alpha, beta)
        nos[i].grau = grau
        nos[i].peso = peso
    for i, no in enumerate(nos):
        if all(no.peso >= nos[j].peso for j in vizinhos[i]):
            CHs.add(i)
    return CHs
def limite_membros(tipo, total):
    if tipo == 'small':
        return max(1, math.ceil(0.05 * total))
    elif tipo == 'medium':
        return max(1, math.ceil(0.15 * total))
    else:
        return max(1, math.ceil(0.30 * total))

def formar_clusters(nos, vizinhos, CHs, max_dist=250):
    total = len(nos)
    clusters = {ch: [] for ch in CHs}
    limites = {ch: limite_membros(nos[ch].tipo, total) for ch in CHs}
    nao_alocados = set(i for i in range(total) if i not in CHs)

    while nao_alocados:
        mudou = False
        for i in list(nao_alocados):
            no = nos[i]
            candidatos = [
                ch for ch in CHs
                if nos[ch].tipo == no.tipo
                and euclideana((no.x, no.y), (nos[ch].x, nos[ch].y)) <= max_dist
                and len(clusters[ch]) < limites[ch]
            ]
            candidatos.sort(key=lambda ch: (-nos[ch].peso, euclideana((no.x, no.y), (nos[ch].x, nos[ch].y))))
            if candidatos:
                clusters[candidatos[0]].append(i)
                nao_alocados.remove(i)
                mudou = True
        if not mudou:
            # Promove só UM nó a CH por vez
            i = nao_alocados.pop()
            CHs.add(i)
            clusters[i] = []
            limites[i] = limite_membros(nos[i].tipo, total)
    return clusters
def aplicar_mst(clusters, nos, E_tx, E_rx, bateria):
    total_edges = []
    for ch, membros in clusters.items():
        todos = [ch] + membros
        G = nx.Graph()
        G.add_nodes_from(todos)
        for i in range(len(todos)):
            for j in range(i + 1, len(todos)):
                u, v = todos[i], todos[j]
                dist = euclideana((nos[u].x, nos[u].y), (nos[v].x, nos[v].y))
                G.add_edge(u, v, weight=dist)

        T = nx.minimum_spanning_tree(G)
        for u, v in T.edges():
            bateria[u] -= E_tx
            bateria[v] -= E_rx
            total_edges.append((u, v))
    return bateria, total_edges
def conectar_CHs_ate_bs(nos, CHs, bs_pos):
    G = nx.Graph()
    ch_list = list(CHs)

    for i in range(len(ch_list)):
        for j in range(i + 1, len(ch_list)):
            u, v = ch_list[i], ch_list[j]
            dist = euclideana((nos[u].x, nos[u].y), (nos[v].x, nos[v].y))
            G.add_edge(u, v, weight=dist)

    BS_ID = 'BS'
    for ch in CHs:
        dist = euclideana((nos[ch].x, nos[ch].y), bs_pos)
        G.add_edge(ch, BS_ID, weight=dist)

    T = nx.minimum_spanning_tree(G)
    interligacoes = [(u, v) for u, v in T.edges() if BS_ID not in (u, v)]
    bs_links = [(u, v) for u, v in T.edges() if BS_ID in (u, v)]

    return interligacoes, bs_links

def visualizar_ucka(nos, CHs, clusters, edges, bs_pos):
    fig, ax = plt.subplots(figsize=(10, 10))

    ch_list = list(clusters.keys())
    cores = plt.colormaps['tab20']
    ch_cor = {ch: cores(i % 20) for i, ch in enumerate(ch_list)}
    formatos = {'small': '^', 'medium': 's', 'large': 'o'}

    # Intra-cluster MSTs
    for ch, membros in clusters.items():
        cor = ch_cor[ch]
        for u, v in edges:
            if u in membros + [ch] and v in membros + [ch]:
                x1, y1 = nos[u].x, nos[u].y
                x2, y2 = nos[v].x, nos[v].y
                ax.plot([x1, x2], [y1, y2], color=cor, linewidth=2)

    # Inter-CH and CH→BS links
    inter, bs_links = conectar_CHs_ate_bs(nos, CHs, bs_pos)

    for u, v in inter:
        x1, y1 = nos[u].x, nos[u].y
        x2, y2 = nos[v].x, nos[v].y
        ax.plot([x1, x2], [y1, y2], color='black', linestyle='dotted', linewidth=1.5)

    for u, v in bs_links:
        ch = u if v == 'BS' else v
        x1, y1 = nos[ch].x, nos[ch].y
        x2, y2 = bs_pos
        ax.plot([x1, x2], [y1, y2], color='black', linestyle='dashed', linewidth=1.5)

    for no in nos:
        if no.indice in CHs:
            ax.scatter(no.x, no.y, c='black', s=200, marker='o', label='CH' if no.indice == list(CHs)[0] else "")
        else:
            dono = next((ch for ch, membros in clusters.items() if no.indice in membros), None)
            cor = ch_cor.get(dono, 'gray')
            ax.scatter(no.x, no.y, c=[cor], s=100, marker=formatos[no.tipo])

    ax.scatter(*bs_pos, c='yellow', s=300, marker='*', label='Base Station')
    ax.set_title('UCKA – Visualização dos Clusters')
    ax.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def calcular_graus(nos, vizinhos):
    for i, viz in enumerate(vizinhos):
        nos[i].grau = len(viz)




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
    

import networkx as nx

def construir_mst_cluster(cluster_nodes, nos):
    """
    Recebe a lista de índices `cluster_nodes` (por exemplo [CH, m1, m2, ...])
    e a lista completa de objetos `nos` (instâncias de No).
    Retorna um dicionário de adjacência representando a MST daquele cluster,
    onde cada nó chave aparece com a lista de vizinhos (grafo não orientado).
    """
    G = nx.Graph()
    # Adiciona cada nó do cluster
    G.add_nodes_from(cluster_nodes)

    # Para cada par dentro do cluster, adiciona aresta ponderada pela distância euclidiana
    for i in range(len(cluster_nodes)):
        for j in range(i + 1, len(cluster_nodes)):
            u = cluster_nodes[i]
            v = cluster_nodes[j]
            dist = euclideana((nos[u].x, nos[u].y), (nos[v].x, nos[v].y))
            G.add_edge(u, v, weight=dist)

    # Obtém MST com networkx
    T = nx.minimum_spanning_tree(G)

    # Monta dicionário de adjacência a partir da MST
    adj = {n: [] for n in cluster_nodes}
    for u, v in T.edges():
        adj[u].append(v)
        adj[v].append(u)
    return adj
def calcular_custo_total(numero_pacotes, distancia_metros):
    """
    Calcula o custo energético total para transmitir e receber múltiplos pacotes LoRaWAN.

    Parâmetros:
    - numero_pacotes: int -> número de pacotes a enviar/receber
    - distancia_metros: float -> distância até o próximo nó em metros

    Retorna:
    - transmissao: energia total de transmissão (em Joules)
    - recepcao: energia total de recepção (em Joules)
    """
    # Parâmetros do rádio LoRa
    V_oper = 3.3              # Tensão de operação em Volts
    I_tx = 0.047              # Corrente de transmissão em Ampères (47 mA)
    I_rx = 0.010              # Corrente de recepção em Ampères (10 mA)

    # Tamanho fixo do pacote em bytes
    tamanho_pacote_bytes = 27

    # Escolha simplificada do Spreading Factor (SF) com base na distância
    if distancia_metros <= 500:
        SF = 7
    elif distancia_metros <= 1000:
        SF = 9
    else:
        SF = 12  # para longas distâncias

    # Taxa de bits efetiva (bps) para 125 kHz BW e CR=4/5
    Rb_SF = {
        7: 5470,
        9: 1460,
        12: 293
    }
    Rb = Rb_SF.get(SF, 293)  # valor padrão SF12

    # Overhead em bits (aproximação para cabeçalho, CRC etc.)
    overhead_bits = 64  # bits adicionais fixos

    # Conversão do tamanho do pacote de bytes para bits
    total_bits_pacote = tamanho_pacote_bytes * 8 + overhead_bits

    # Tempo no ar (segundos) para um pacote
    tempo_no_ar_s = total_bits_pacote / Rb

    # Potências em Watts
    P_tx = V_oper * I_tx
    P_rx = V_oper * I_rx

    # Energia por pacote
    energia_tx_pacote = P_tx * tempo_no_ar_s
    energia_rx_pacote = P_rx * tempo_no_ar_s

    # Energia total (todos os pacotes)
    transmissao = energia_tx_pacote * numero_pacotes
    recepcao = energia_rx_pacote * numero_pacotes

    return transmissao, recepcao
def _dfs_contagem_pacotes(u, parent, adj, contagem):
    """
    DFS auxiliar para calcular contagem de pacotes na subárvore enraizada em `u`.
    - `adj` é o dicionário de adjacência do MST daquele cluster.
    - `contagem` é um dict que vai mapear nó -> número total de pacotes que esse nó deve
      enviar ao seu pai (incluindo o próprio).
    """
    total_sub = 1  # cada nó gera 1 pacote próprio
    for v in adj[u]:
        if v == parent:
            continue
        _dfs_contagem_pacotes(v, u, adj, contagem)
        # contagem[v] é quantos pacotes o filho v envia ao u,
        # todos esses pacotes devem ser recebidos por u.
        total_sub += contagem[v]
    contagem[u] = total_sub


def encaminhar_dados_cluster(cluster_nodes, CH, nos, bateria, bs_pos):
    """
    - cluster_nodes: lista de índices de nós que compõem este cluster (inclui CH).
    - CH: índice do Cluster Head atual.
    - nos: lista com instâncias de No para *TODOS* os nós da rede.
    - bateria: lista de floats (bateria[i] atual do nó i). 
               Será atualizada in place.
    - bs_pos: tupla (x_bs, y_bs).

    Retorna: lista de nós (índices) cujo bateria[i] <= 0 após o cálculo.
    """
    # 1) Monta a MST do cluster
    adj = construir_mst_cluster(cluster_nodes, nos)

    # 2) Calcula quantos pacotes cada nó envia ao pai (contagem[u])
    contagem = {}
    _dfs_contagem_pacotes(CH, None, adj, contagem)

    mortos = []

    # 3) Processa todos os nós exceto CH
    for u in cluster_nodes:
        if u == CH:
            continue
        # Descobre o pai de u na MST (há exatamente um caminho), basta verificar adj[u] e filtrar quem não é filho.
        # Mas para saber quem é pai, faremos: qualquer vizinho de u que seja parte do caminho para CH. 
        # Como a MST é pequena, podemos achar o pai procurando aquele vizinho cujo contagem[viz] > contagem[u], 
        # mas isso falha em casos onde filho e pai tenham contagem igual (improvável). 
        # Em vez disso, vamos fazer uma micro-DFS desde CH para achar pai[u].

        # Para simplificar, criamos um dicionário parent=u→pai.
        parent = _encontrar_pai_em_mst(u, CH, adj)

        # 3a) Número de pacotes que u **recebe** dos filhos diretos:
        pacotes_recebidos = 0
        for v in adj[u]:
            if v != parent:
                # v é filho de u
                pacotes_recebidos += contagem[v]

        # 3b) Cálculo de energia:
        dist_tx = euclideana((nos[u].x, nos[u].y), (nos[parent].x, nos[parent].y))
        # custo_rx: energia para receber todos pacotes_recebidos (distance não muda a potência de RX, mas usaremos mesma SF do tx).
        _, energia_rx_total = calcular_custo_total(pacotes_recebidos, dist_tx)
        # custo_tx: energia para transmitir contagem[u] pacotes a parent
        energia_tx_total, _ = calcular_custo_total(contagem[u], dist_tx)

        bateria[u] -= (energia_rx_total + energia_tx_total)
        if bateria[u] <= 0:
            mortos.append(u)

    # 4) Processa o CH: ele recebe dos filhos diretos e envia tudo à BS
    pacotes_de_entrada = 0
    for v in adj[CH]:
        # todo vizinho v de CH é filho, pois CH é raiz
        pacotes_de_entrada += contagem[v]

    # 4a) Custo de recepção no CH:
    # Para cada filho v, seus pacotes chegam em CH; 
    # vamos somar o custo de recepção de cada contagem[v] em dist = dist(v, CH).
    energia_rx_CH = 0.0
    for v in adj[CH]:
        dist_v_ch = euclideana((nos[v].x, nos[v].y), (nos[CH].x, nos[CH].y))
        # cada v envia contagem[v] pacotes → CH recebe contagem[v] pacotes dele
        _, erx = calcular_custo_total(contagem[v], dist_v_ch)
        energia_rx_CH += erx

    # 4b) Custo de transmissão CH → BS: 
    # o CH atualmente tem que transmitir contagem[CH] pacotes para a BS
    total_pacotes_CH = contagem[CH]
    dist_ch_bs = euclideana((nos[CH].x, nos[CH].y), bs_pos)
    etx_ch, _ = calcular_custo_total(total_pacotes_CH, dist_ch_bs)

    bateria[CH] -= (energia_rx_CH + etx_ch)
    if bateria[CH] <= 0:
        mortos.append(CH)

    return mortos


def _encontrar_pai_em_mst(u, raiz, adj):
    """
    Retorna o pai de `u` na árvore enraizada em `raiz` (CH), usando um BFS/DFS curto.
    Como a MST é pequena, podemos fazer uma busca simples.
    """
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

def remover_nos_mortos(clusters, mortos):
    """
    - clusters: dict {CH: [membros, ...], ...}
    - mortos: lista (ou set) de índices de nós que acabaram de morrer.

    Remove cada nó morto:
      a) Se estiver em members de algum cluster, remove de lista.
      b) Se for CH (chave do dict), remove toda a chave do dict e
         coloca seus membros (que não tenham morrido) em uma lista
         temporária para reeleição posterior.

    Retorna dois itens:
      - novos_clusters: dict {novo_CH: [membros..., ...], ...} mas *sem* escolher CH aqui;
        somente removeu nós mortos. Se um CH morreu, teremos uma entrada
        em um dicionário auxiliar `cluster_com_ch_vazio[antigo_CH] = membros_restantes`.
      - cluster_com_ch_vazio: dict {antigo_CH: membros_restantes}, para sabermos
        quais clusters perderam CH e precisam de reeleição.
    """
    cluster_com_ch_vazio = {}
    novos = {}

    for ch, membros in clusters.items():
        # Se o CH morreu:
        if ch in mortos:
            # todos os membros sobreviventes desse cluster agora ficam
            # à espera de reeleição de CH.
            sobraram = [m for m in membros if m not in mortos]
            if sobraram:
                cluster_com_ch_vazio[ch] = sobraram
            # Se não sobrou ninguém, simplesmente esse cluster desaparece.
        else:
            # CH não morreu. Filtra membros mortos:
            membros_sobreviventes = [m for m in membros if m not in mortos]
            novos[ch] = membros_sobreviventes

    return novos, cluster_com_ch_vazio

def reeleicao_CHs_clusters(clusters, nos, viz, alpha=0.5, beta=0.5):
    """
    - clusters: dict {CH_antigo: [membros, ...], ...}, já sem nós mortos.
    - nos: lista de objetos No, cada um com atributo `.energia` e `.grau` atualizados.
    - viz: lista de listas de vizinhos fixos (ainda podem conter índices mortos,
           mas só nos interessa o atributo grau que já foi calculado).
    - alpha, beta: parâmetros de peso (mesmos da função eleger_CHs original).

    Para cada cluster:
      a) Monta lista_de_candidatos = [CH_antigo] + membros.
      b) Recalcula peso_i = alpha * nos[i].energia + beta * nos[i].grau.
      c) Escolhe i_max onde peso_i é máximo. Esse será o novo CH.
    Retorna:
      - novos_clusters: dict {novo_CH: [todos os outros membros do cluster], ...}
      - CHs_anteriores: set(dos CHs_antigos que sobreviveram)  (para comparação)
      - CHs_novos: set(dos novos CHs) 
    """
    CHs_anteriores = set(clusters.keys())
    CHs_novos = set()
    novos_clusters = {}

    for ch_antigo, membros in clusters.items():
        candidatos = [ch_antigo] + membros
        # Filtra apenas os que não estão mortos e ainda existem em `nos` (por segurança)
        candidatos_vivos = [i for i in candidatos if nos[i].energia > 0]

        # Recalcula peso para cada candidato
        best = None
        maior_peso = -float('inf')
        for i in candidatos_vivos:
            grau_i = len(viz[i])  # ou use nos[i].grau se já atualizado antes
            peso_i = alpha * nos[i].energia + beta * grau_i
            if peso_i > maior_peso:
                maior_peso = peso_i
                best = i

        if best is None:
            # todo cluster morreu (nenhum nó vivo). Então não incluímos no dict final.
            continue

        # O novo CH é `best`; os demais continuam membros
        CHs_novos.add(best)
        outros = [i for i in candidatos_vivos if i != best]
        novos_clusters[best] = outros

    return novos_clusters, CHs_anteriores, CHs_novos


def checar_porcentagem_mudanca(CHs_anteriores, CHs_novos, limiar=0.20):
    """
    - CHs_anteriores: set de índices de CH da rodada passada
    - CHs_novos: set de índices de CH da rodada atual
    - limiar: fração mínima de CHs que mudaram para disparar a plotagem

    Retorna: (frac_mudanca, mudou_flag):
      - frac_mudanca = (#CHs que mudaram) / (#CHs_anteriores) 
      - mudou_flag = True se frac_mudanca >= limiar
    """
    if not CHs_anteriores:
        return 1.0, True  # na primeira rodada, considere mudança total.

    iguais = CHs_anteriores.intersection(CHs_novos)
    total_anteriores = len(CHs_anteriores)
    total_diferentes = total_anteriores - len(iguais)
    frac = total_diferentes / total_anteriores
    return frac, (frac >= limiar)

def rodada_multihop(clusters, nos, viz, bateria, bs_pos, alpha=0.5, beta=0.5):
    """
    Faz uma iteração completa de multihop:
      1) Encaminha dados em cada cluster e atualiza baterias
      2) Remove nós mortos, mas se o CH morrer, mantém membros vivos para reeleição
      3) Sincroniza nos[i].energia = bateria[i]
      4) Reeleição de CH em todos os clusters (inclusive aqueles que perderam o CH)
    
    Parâmetros:
    - clusters: dict {CH_antigo: [membros, ...], ...}
    - nos: lista de objetos No
    - viz: lista de listas de vizinhos fixos
    - bateria: lista de floats com bateria atual (será atualizada in-place)
    - bs_pos: tupla (x_bs, y_bs)
    - alpha, beta: pesos para cálculo de peso na eleição de CH

    Retorna:
    - novos_clusters: dict {novo_CH: [membros, ...], ...}
    - CHs_anteriores: set de CHs antes da reeleição
    - CHs_novos: set de CHs depois da reeleição
    - mortos_rodada: lista de nós que morreram nesta rodada
    """
    todos_mortos_rodada = set()

    # 1) Encaminhamento multihop em cada cluster
    for ch, membros in list(clusters.items()):
        cluster_nodes = [ch] + membros
        mortos_cluster = encaminhar_dados_cluster(cluster_nodes, ch, nos, bateria, bs_pos)
        todos_mortos_rodada.update(mortos_cluster)

    # 2) Remove nós mortos, mas preserva membros se o CH morrer
    clusters_sem_mortos, cluster_com_ch_vazio = remover_nos_mortos(clusters, todos_mortos_rodada)
    # Aqui, clusters_sem_mortos já contém apenas clusters cujo CH sobreviveu (com membros filtrados).
    # cluster_com_ch_vazio contém ch antigos que morreram, mas com uma lista de membros ainda vivos.
    # Vamos unir ambos para permitir a reeleição:
    clusters_para_reeleicao = {}
    # a) Clusters onde o CH sobreviveu (chaves de clusters_sem_mortos)
    for ch_surv, membros_surv in clusters_sem_mortos.items():
        clusters_para_reeleicao[ch_surv] = membros_surv
    # b) Clusters onde o CH morreu, mas membros ainda vivem
    for ch_dead, membros_restantes in cluster_com_ch_vazio.items():
        clusters_para_reeleicao[ch_dead] = membros_restantes

    # 3) Sincroniza a energia armazenada no objeto 'No' com a lista 'bateria'
    for i in range(len(nos)):
        nos[i].energia = bateria[i]

    # 4) Reeleição de CH em todos os clusters de clusters_para_reeleicao
    novos_clusters, CHs_anteriores, CHs_novos = reeleicao_CHs_clusters(
        clusters_para_reeleicao, nos, viz, alpha, beta
    )

    return novos_clusters, CHs_anteriores, CHs_novos, list(todos_mortos_rodada)


def coletar_arestas_intracluster(clusters, nos):
    """
    Para cada cluster em `clusters`, chama construir_mst_cluster([ch] + membros)
    e consegue todas as arestas da MST desse cluster. Retorna uma lista única
    de tuplas (u, v), com u < v, para evitar duplicatas.
    """
    todas_edges = set()
    for ch, membros in clusters.items():
        cluster_nodes = [ch] + membros
        adj = construir_mst_cluster(cluster_nodes, nos)
        # adj: dicionário {node: [vizinhos_na_MST], ...}
        for u, vizs in adj.items():
            for v in vizs:
                # garante (menor,maior) para manter consistência e não duplicar
                a, b = (u, v) if u < v else (v, u)
                todas_edges.add((a, b))
    # volta como lista de tuplas
    return list(todas_edges)


def main():
    path = 'Cenário 4 - Rede 100.txt'
    coords = ler_coords_arquivo(path)
    bs_pos = (400, 200)
    energia_inicial = 2700.0
    tx_range = 150
    max_rounds = 50
    alpha, beta = 0.5, 0.5

    # 1) Cria lista de instâncias No e rede de vizinhos (grafo original)
    nos = inicializar_nos(coords, energia_inicial, bs_pos)
    viz = construir_vizinhos(nos, tx_range)
    calcular_graus(nos, viz)
    bateria = [energia_inicial for _ in nos]

    # 2) Eleição inicial de CHs e formação de clusters
    CHs_iniciais = eleger_CHs(nos, viz, alpha, beta)
    clusters = formar_clusters(nos, viz, CHs_iniciais)

    # 3) Plota estado inicial (antes da rodada 1)
    arestas_iniciais = coletar_arestas_intracluster(clusters, nos)
    print("🔷 Estado inicial antes da rodada 1:")
    visualizar_ucka(nos, set(clusters.keys()), clusters, arestas_iniciais, bs_pos)

    prev_CHs = set(CHs_iniciais)
    mortinhos_acumulado = set()

    # 4) Loop de rodadas
    for rodada in range(1, max_rounds + 1):
        print(f"\n🔁 Rodada {rodada}")

        # 4.1) Atualiza peso de cada nó (energia mudou em rodadas anteriores)
        for i, no in enumerate(nos):
            no.peso = alpha * no.energia + beta * len(viz[i])

        # 4.2) Executa multihop, remoção de mortos e reeleição
        clusters, CHs_anteriores, CHs_novos, mortos_rodada = rodada_multihop(
            clusters, nos, viz, bateria, bs_pos, alpha, beta
        )

        mortinhos_acumulado.update(mortos_rodada)

        print(f"→ CHs antigos: {sorted(CHs_anteriores)}")
        print(f"→ CHs novos:   {sorted(CHs_novos)}")
        print(f"→ Nós que morreram nesta rodada: {sorted(mortos_rodada)}")

        # 4.3) Se não há mais clusters (todos morreram), encerra
        if not clusters:
            print("✅ Todos os nós morreram ou nenhum cluster restou. Fim da simulação.")
            break

        # 4.4) Calcula fração de CHs que mudaram e, se ≥ 20 %, plota figura
        frac, disparar = checar_porcentagem_mudanca(CHs_anteriores, CHs_novos, limiar=0.20)
        print(f"   * Fração de CHs que mudaram: {frac:.2%}")

        if disparar:
            # Recolhe arestas intracluster para desenhar
            arestas_atual = coletar_arestas_intracluster(clusters, nos)
            print(f"   ! Mais de 20% dos CHs mudaram → gerando figura na rodada {rodada}")
            visualizar_ucka(nos, CHs_novos, clusters, arestas_atual, bs_pos)
            print(f"   * Iteração atual: {rodada}")
            print(f"   * Total de nós mortos até agora: {sorted(mortinhos_acumulado)}")

        prev_CHs = CHs_novos

    # 5) Opcional: plota estado final caso ainda existam clusters
    if clusters:
        ultimos_CHs = set(clusters.keys())
        print(f"\n✅ Fim após {rodada} rodadas. CHs finais: {sorted(ultimos_CHs)}")
        arestas_finais = coletar_arestas_intracluster(clusters, nos)
        visualizar_ucka(nos, ultimos_CHs, clusters, arestas_finais, bs_pos)



if __name__ == "__main__":
    main()
