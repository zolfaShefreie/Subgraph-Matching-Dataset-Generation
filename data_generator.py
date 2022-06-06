import networkx as nx


class GraphPlusDict(nx.Graph):

    def graph_to_dict(self) -> dict:
        """
        convert graph to dictionary format
        :return: dict graph info
        """
        return {
            "nodes": {item[0]: item[1] for item in list(self.nodes(data=True))},
            "edges": {(item[0], item[1]): item[2] for item in list(self.edges(data=True))},
        }


class SubGraphMatchingElement:

    def __int__(self, source_graph: GraphPlusDict, query_graph: GraphPlusDict, label):
        self.source_graph = source_graph
        self.query_graph = query_graph
        self.label = label

    def to_dict(self) -> dict:
        """
        convert graph pair element to dictionary format
        :return: dict pair format
        """
        return {
            "source_graph": self.source_graph.graph_to_dict(),
            "query_graph": self.query_graph.graph_to_dict(),
            "label": self.label
        }


class SubGraphDatasetGenerator:

    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        