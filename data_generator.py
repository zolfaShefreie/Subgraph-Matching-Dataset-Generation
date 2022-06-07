import networkx as nx
from os import listdir
from os.path import isfile, join
import random
import itertools


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
    """
    contain a pair of graph with a label
    """

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
    """
    this class generate dataset for subgraph matching problem.
    stages for create dataset:
        1. read main graph dataset
        2. create graphs one by one
        3. create subgraphs that match or doesn't match the graph
        4. save the elements of new dataset
    """

    NODE_NOISE_THRESHOLD = 0.2
    EDGE_NOISE_THRESHOLD = 0.3
    ATTRIBUTE_NOISE_THRESHOLD = 0.7

    @classmethod
    def generate(cls, dataset_dir: str):
        files = cls._get_dataset_open_files(dataset_dir)
        info_dict = cls._scan_to_get_info(files)
        graph = cls._create_graph_from_file(files)
        while len(list(graph.nodes)) != 0:
            graph = cls._create_graph_from_file(files)

    @classmethod
    def _get_dataset_open_files(cls, dataset_dir: str) -> dict:
        """
        get all files of dir and open files
        :param dataset_dir: path of dataset
        :return:
        """
        files = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
        dataset_name = dataset_dir.split("/")[-1]
        return {file_name[:-4].split(dataset_name)[-1][1:]: {'file': open(dataset_dir + "/" + file_name, mode='r',
                                                                          encoding="utf-8"),
                                                             'position': 0,
                                                             'line': 0}
                for file_name in files}

    @classmethod
    def _scan_to_get_info(cls, files: dict) -> dict:
        """
        scan the files and get some info
        :param files:
        :return: info dict
        """
        info = dict()
        if 'node_labels' in files.keys():
            lines = files['node_labels']['file'].readlines()
            label_array = [line.strip('\n')[0] for line in lines if line != "\n"]
            info['node'] = {'label': {'max': max(label_array), 'min': min(label_array)}}
            files['node_labels']['file'].seek(0)

        if 'edge_labels' in files.keys():
            lines = files['edge_labels']['file'].readlines()
            label_array = [line.strip('\n')[0] for line in lines if line != "\n"]
            info['edge'] = {'label': {'max': max(label_array), 'min': min(label_array)}}
            files['edge_labels']['file'].seek(0)

        if 'node_attributes' in files.keys():
            attr_dict = dict()
            lines = files['node_attributes']['file'].readlines()
            for line in lines:
                if line != "\n":
                    attrs = line.strip('\n')[0].strip[', ']
                    for i, attr in enumerate(attrs):
                        attr_dict[f"attr_{i}"] = attr_dict.get(f"attr_{i}", []).append(attr)
            info['node'].update({key: {"max": max(value), "min": min(value)} for key, value in attr_dict.items()})
            files['node_attributes']['file'].seek(0)

        if 'edge_attributes' in files.keys():
            attr_dict = dict()
            lines = files['edge_attributes']['file'].readlines()
            for line in lines:
                if line != "\n":
                    attrs = line.strip('\n')[0].strip[', ']
                    for i, attr in enumerate(attrs):
                        attr_dict[f"attr_{i}"] = attr_dict.get(f"attr_{i}", []).append(attr)
            info['edge'].update({key: {"max": max(value), "min": min(value)} for key, value in attr_dict.items()})
            files['edge_attributes']['file'].seek(0)

        return info

    @classmethod
    def _create_graph_from_file(cls, files: dict) -> GraphPlusDict:
        """
        create graph
        :param files:
            graph_indicator => add nodes, if exist: node_labels, node_attributes => node feature
            A => add edge, if exist: edge_labels, edge_attributes => edge_feature
        :return: graph
        """
        graph = GraphPlusDict()

        # adding nodes and their attributes
        last_graph_id = -1

        while True:
            line = files['graph_indicator']['file'].readline()
            if line == "\n":
                return graph

            if last_graph_id != -1 and last_graph_id == line.split('\n')[0]:
                files['graph_indicator']['file'].seek(files['graph_indicator']['position'], 0)
                break
            last_graph_id = line.split('\n')[0]
            files['graph_indicator']['position'] = files['graph_indicator']['file'].tell()
            files['graph_indicator']['line'] = files['graph_indicator']['line'] + 1

            node_atrrs = dict()

            if 'node_labels' in files.keys():
                node_atrrs['label'] = int(files['node_labels']['file'].readline().split('\n')[0])
                files['node_labels']['position'] = files['node_labels']['file'].tell()
                files['node_labels']['line'] = files['node_labels']['line'] + 1

            if 'node_attributes' in files.keys():
                node_attributes = files['node_attributes']['file'].readline().split('\n')[0].split(', ')
                node_atrrs.update({f'attr{index+1}': float(attr) for index, attr in enumerate(node_attributes)})
                files['node_attributes']['position'] = files['node_attributes']['file'].tell()
                files['node_attributes']['line'] = files['node_attributes']['line'] + 1

            graph.add_node(files['graph_indicator']['line'], **node_atrrs)

        # adding edges and their attributes
        graph_nodes = set(list(graph.nodes))
        while True:
            line = files['A']['file'].readline()
            if line == "\n":
                return graph

            node_ids = [int(node_id) for node_id in line.split('\n').split(', ')]
            if len(set(node_ids) - graph_nodes) > 0:
                files['A']['file'].seek(files['A']['position'], 0)
                break
            files['A']['position'] = files['A']['file'].tell()
            files['A']['line'] = files['A']['line'] + 1

            edge_attrs = dict()
            if 'edge_labels' in files.keys():
                edge_attrs['label'] = int(files['edge_labels']['file'].readline().split('\n')[0])
                files['edge_labels']['position'] = files['edge_labels']['file'].tell()
                files['edge_labels']['line'] = files['edge_labels']['line'] + 1

            if 'edge_attributes' in files.keys():
                edge_attributes = files['edge_attributes']['file'].readline().split('\n')[0].split(', ')
                edge_attrs.update({f'attr{index+1}': float(attr) for index, attr in enumerate(edge_attributes)})
                files['edge_attributes']['position'] = files['edge_attributes']['file'].tell()
                files['edge_attributes']['line'] = files['edge_attributes']['line'] + 1

            graph.add_edge(node_ids[0], node_ids[1], **edge_attrs)

        return graph

    @classmethod
    def _get_random_subgraph(cls, graph: GraphPlusDict) -> GraphPlusDict:
        """
        return a random subgraph
        :param graph:
        :return:
        """
        graph_nodes = set(list(graph.nodes))
        random_len_subgraph = random.randint(int(len(graph_nodes)/2)+1, len(graph_nodes))
        combinations_subgraph_nodes = list(itertools.combinations(graph_nodes, random_len_subgraph))
        subgraph_nodes = combinations_subgraph_nodes[random.randint(0, len(combinations_subgraph_nodes))]
        return graph.subgraph(subgraph_nodes)

    @classmethod
    def _add_noise_to_graph(cls, graph: GraphPlusDict) -> GraphPlusDict:
        pass



