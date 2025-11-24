import networkx as nx

from graph_builder import GraphBuilder

class SpatialBuilder(GraphBuilder):
    """
    A builder for an N x N spatial grid.
    - Each node is uniquely identified by (row, col) or a string name.
    - For each pair of distinct nodes A,B:
        * If rowA < rowB, A -> B is "NORTH_OF", B -> A is "SOUTH_OF".
        * Else if rowA > rowB, A -> B is "SOUTH_OF", B -> A is "NORTH_OF".
        * Else (same row): compare colA, colB for east/west relationships.
    """

    RELATIONSHIP_TYPES = [
        "NO_RELATION",  # 0
        "NORTH_OF",     # 1
        "SOUTH_OF",     # 2
        "EAST_OF",      # 3
        "WEST_OF",      # 4
    ]

    def __init__(self, grid_size=3):
        super().__init__()
        self.grid_size = grid_size
        # Let the environment/GCN know about these relationships
        self.relation_types = self.RELATIONSHIP_TYPES
        self.test_relation_types = self.RELATIONSHIP_TYPES
        self.rel_to_id = {r: i for i, r in enumerate(self.RELATIONSHIP_TYPES)}

    def build_graph(self):
        """
        Create an N x N set of nodes, each named "rXcY" or (row, col).
        Then for every pair of distinct nodes, add exactly 2 directed edges
        with complementary relationships (e.g. A->B is NORTH_OF => B->A is SOUTH_OF).
        """
        G = nx.MultiDiGraph()

        # 1) Create node IDs
        node_names = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node_names.append(f"r{r}c{c}")

        # Add all nodes to the graph
        for name in node_names:
            G.add_node(name)

        # A helper function to parse row,col from "r2c1"
        def parse_coords(node_str):
            # node_str is "r{r}c{c}"
            # e.g. "r2c1" => (2, 1)
            parts = node_str.split("c")
            left = parts[0]  # e.g. "r2"
            row = int(left[1:])  # after 'r'
            col = int(parts[1])
            return (row, col)

        # 2) For every pair of distinct nodes, add edges
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if i == j:
                    continue
                A = node_names[i]
                B = node_names[j]
                rA, cA = parse_coords(A)
                rB, cB = parse_coords(B)

                # Determine the single relationship from A->B
                if rA < rB:
                    relAB = "NORTH_OF"
                    relBA = "SOUTH_OF"
                elif rA > rB:
                    relAB = "SOUTH_OF"
                    relBA = "NORTH_OF"
                else:
                    # same row => compare columns
                    if cA < cB:
                        relAB = "WEST_OF"
                        relBA = "EAST_OF"
                    else:
                        relAB = "EAST_OF"
                        relBA = "WEST_OF"

                # Add edges to the MultiDiGraph
                G.add_edge(A, B, relationship=relAB)
                # The opposite direction
                G.add_edge(B, A, relationship=relBA)

        return G

    def sample_observations(self, G, n=1):
        """
        Randomly pick n edges from G and return (src, dst, relationship).
        Skipping "NO_RELATION" since we only assigned the 4 directional ones.
        """
        edges = list(G.edges(keys=True, data=True))
        if not edges:
            return []

        obs = []
        for _ in range(n):
            u, v, key, data = self._rng.choice(edges)
            rel = data.get("relationship", None)
            if rel:
                obs.append((u, v, rel))
        return obs
