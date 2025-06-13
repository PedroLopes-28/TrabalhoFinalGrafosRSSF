import random
import math

# Disjoint Set Union (Union-Find)
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

# Calculate Euclidean distance
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Generate sensor positions
def generate_sensors(n, area_size, conn_thresh=150, target_avg=100, tolerance=10):
    while True:
        sensors = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n)]
        edges = []
        uf = UnionFind(n)

        # Create edges for sensors within threshold
        for i in range(n):
            for j in range(i+1, n):
                d = distance(sensors[i], sensors[j])
                if d <= conn_thresh:
                    edges.append(d)
                    uf.union(i, j)

        # Verify connectivity (optional: ensure at least one large cluster)
        clusters = set(uf.find(i) for i in range(n))

        if len(edges) == 0:
            continue

        avg_dist = sum(edges) / len(edges)

        if abs(avg_dist - target_avg) <= tolerance:
            return sensors, avg_dist, len(clusters)

# Save sensors to file
def save_to_txt(filename, sensors):
    with open(filename, 'w') as f:
        f.write(f"{len(sensors)}\n")
        for x, y in sensors:
            f.write(f"{x}, {y}\n")

# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate RSF sensor dataset") #muda o default pra mudar os dados gerados
    parser.add_argument('--n', type=int, default=1000, help='Number of vertices (sensors)') #aqui é o numero de nós
    parser.add_argument('--area', type=float, default=1000.0, help='Size of the square area (meters)') #aqui é a area total 
    parser.add_argument('--output', type=str, default='sensores.txt', help='Output file name')#aqui é o nome do arquivo
    args = parser.parse_args()

    sensors, avg, cluster_count = generate_sensors(n=args.n, area_size=args.area)
    save_to_txt(args.output, sensors)

    print(f"Generated {args.n} sensors in {args.area}x{args.area} m² area")
    print(f"Average connection distance: {avg:.2f} m, clusters formed: {cluster_count}")
