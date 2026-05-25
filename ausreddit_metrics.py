import networkx as nx
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class AusredditMetrics:
    """Compute per-submission conversation metrics from a Reddit conversation graph.

    Designed to work with graphs produced by Reddit_trees.tree_graph_and_adj_list(),
    which adds each submission as the root node of its conversation tree. Each
    connected component in that graph corresponds to exactly one submission, so
    analyze_conversation_graphs() correctly identifies and measures each tree.

    Typical usage:
        trees = Reddit_trees()
        G, adj = trees.tree_graph_and_adj_list(comments_df, submissions_df)
        metrics = AusredditMetrics()
        df = metrics.analyze_conversation_graphs(G)
    """

    def __init__(self):
        pass

    def analyze_conversation_graphs(self, G: nx.DiGraph) -> pd.DataFrame:
        """Analyze metrics for each submission tree in a combined conversation graph.

        Splits the graph into per-submission components using connected-component
        analysis on the undirected projection, then computes metrics for each.
        This works correctly when the graph was built by tree_graph_and_adj_list()
        with a submissions_df argument, so that each submission node is present
        as the root of its tree.

        Parameters:
            G: Combined directed graph from Reddit_trees.tree_graph_and_adj_list().

        Returns:
            DataFrame indexed by submission ID, one row per submission tree.
        """
        metrics = []
        for component in nx.connected_components(G.to_undirected()):
            subgraph = G.subgraph(component).copy()

            # Identify the submission node to use as the stable graph_id.
            submission_nodes = [
                n for n, d in subgraph.nodes(data=True)
                if d.get('node_type') == 'submission'
            ]
            graph_id = submission_nodes[0] if submission_nodes else None
            metrics.append(self.calculate_subgraph_metrics(subgraph, graph_id))

        df = pd.DataFrame(metrics)

        column_order = [
            'graph_id',
            'num_comments',
            'num_nodes',
            'num_edges',
            'longest_path_length',
            'average_path_length',
            'num_branches',
            'num_endpoints',
            'total_duration',
            'shortest_response_time',
            'longest_response_time',
            'average_response_time',
        ]

        if 'warning' in df.columns:
            column_order.append('warning')

        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns].set_index('graph_id')

        return df

    def calculate_subgraph_metrics(self, G: nx.DiGraph, graph_id: Optional[str]) -> Dict:
        """Calculate metrics for a single submission conversation tree.

        Parameters:
            G: Directed subgraph for one submission and all its comments.
            graph_id: Submission node ID used as the row identifier in the output.

        Returns:
            Dict of metrics for this tree.
        """
        submission_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get('node_type') == 'submission'
        ]

        metrics = {
            'graph_id': graph_id,
            'num_nodes': G.number_of_nodes(),
            'num_comments': G.number_of_nodes() - len(submission_nodes),
            'num_edges': G.number_of_edges(),
            'warning': None,
        }

        # Root is the submission node; fall back to any in-degree-0 node.
        if submission_nodes:
            root = submission_nodes[0]
        else:
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if not roots:
                raise ValueError(f"No root node found in graph {graph_id}")
            root = roots[0]
            if len(roots) > 1:
                metrics['warning'] = f"Multiple roots found: {roots}"

        longest_path = self.find_longest_path(G, root)
        metrics['longest_path_length'] = len(longest_path) - 1

        path_lengths = self.calculate_path_lengths(G, root)
        metrics['average_path_length'] = int(np.mean(list(path_lengths.values())))

        metrics.update(self.calculate_branch_metrics(G))

        time_metrics = self.calculate_time_metrics(G)
        if time_metrics:
            metrics.update(time_metrics)

        return metrics

    def find_longest_path(self, G: nx.DiGraph, root) -> List:
        """Find the longest directed path from root to any leaf node.

        Parameters:
            G: Directed graph.
            root: Root node ID.

        Returns:
            Ordered list of node IDs on the longest path.
        """
        def dfs(node, path):
            if G.out_degree(node) == 0:
                return path
            longest = path
            for neighbor in G.successors(node):
                candidate = dfs(neighbor, path + [neighbor])
                if len(candidate) > len(longest):
                    longest = candidate
            return longest

        return dfs(root, [root])

    def calculate_path_lengths(self, G: nx.DiGraph, root) -> Dict:
        """Calculate directed path lengths from root to all reachable nodes.

        Parameters:
            G: Directed graph.
            root: Root node ID.

        Returns:
            Dict mapping node ID to depth from root.
        """
        path_lengths = {}

        def dfs(node, length):
            path_lengths[node] = length
            for neighbor in G.successors(node):
                if neighbor not in path_lengths:
                    dfs(neighbor, length + 1)

        dfs(root, 0)
        return path_lengths

    def calculate_branch_metrics(self, G: nx.DiGraph) -> Dict:
        """Calculate branching metrics: branch count and endpoint count.

        Parameters:
            G: Directed graph.

        Returns:
            Dict with num_branches (nodes with >1 child) and num_endpoints (leaf nodes).
        """
        outdegrees = [int(d) for _, d in G.out_degree()]
        return {
            'num_branches': sum(1 for d in outdegrees if d > 1),
            'num_endpoints': sum(1 for d in outdegrees if d == 0),
        }

    def calculate_time_metrics(self, G: nx.DiGraph) -> Optional[Dict]:
        """Calculate time-based metrics for a conversation tree.

        Requires all nodes to have a time_created attribute. Returns None if
        timestamps are missing or malformed.

        Parameters:
            G: Directed graph with time_created node attributes.

        Returns:
            Dict of formatted time metrics, or None if timestamps are unavailable.
        """
        try:
            timestamps = {}
            for node, data in G.nodes(data=True):
                if 'time_created' not in data:
                    return None
                t = data['time_created']
                timestamps[node] = t if isinstance(t, datetime) else pd.to_datetime(t)

            if len(timestamps) != G.number_of_nodes():
                return None

            total_duration = max(timestamps.values()) - min(timestamps.values())

            response_times = []
            for src, tgt in G.edges():
                delta = (timestamps[tgt] - timestamps[src]).total_seconds()
                if delta >= 0:
                    response_times.append(delta)

            if not response_times:
                return None

            def fmt(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            return {
                'total_duration': fmt(total_duration.total_seconds()),
                'shortest_response_time': fmt(min(response_times)),
                'longest_response_time': fmt(max(response_times)),
                'average_response_time': fmt(np.mean(response_times)),
            }

        except (KeyError, AttributeError) as e:
            print(f"Error calculating time metrics: {e}")
            return None
