import networkx as nx
from os import listdir
from os.path import isfile, join
import random
import itertools
import numpy as np


class GraphPlusDict(nx.Graph):
    # all of graphs of datasets are undirected so we use nx.Graph

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
    DELETE_EDGE_P = 0.6
    CHANGE_LABEL_P = 0.4
    ADD_EDGE = 0.3
    ADD_NODE = 0.2
    ADD_NODE_THRESHOLD = 1

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
        random_len_subgraph = random.randint(int(len(graph_nodes)/2)+1, len(graph_nodes) - cls.ADD_NODE_THRESHOLD)
        combinations_subgraph_nodes = list(itertools.combinations(graph_nodes, random_len_subgraph))
        subgraph_nodes = combinations_subgraph_nodes[random.randint(0, len(combinations_subgraph_nodes))]
        return graph.subgraph(subgraph_nodes)

    @classmethod
    def _add_noise_to_graph(cls, graph: GraphPlusDict) -> GraphPlusDict:
        """
        add noise to graph attributes
            noise: 0  - .1 to attribute
                    delete 1 edge
        :param graph:
        :return:
        """
        # TODO change this limitaion based on number of graph.edges
        max_delete_edge = 1 if len(list(graph.edges)) >= 3 else 0

        # add noise to node
        nodes_info = list(graph.nodes(data=True))
        node_new_data = dict()
        for node_attr in nodes_info:
            if random.random() < cls.NODE_NOISE_THRESHOLD:
                # TODO: change max random based on data
                new_attr = node_attr[1].values() + np.random.normal(0, .1, np.array(node_attr[1].values()))
                node_new_data[node_attr[0]] = {list(node_attr[1].keys())[i]: new_attr[i] for i in range(len(new_attr))
                                               if list(node_attr[1].keys())[i] != "label"}
            else:
                node_new_data[node_attr[0]] = node_attr[1]

        # add noise to edge
        edges_info = list(graph.edges(data=True))
        edge_new_data = dict()
        for edge_attr in edges_info:
            if random.random() < cls.DELETE_EDGE_P and max_delete_edge != 0 and \
                    cls._allow_delete_edge(graph, (edge_attr[0], edge_attr[1])):
                graph.remove_edge(edge_attr[0], edge_attr[1])
                max_delete_edge -= 1

            elif random.random() < cls.EDGE_NOISE_THRESHOLD:
                # TODO: change max random based on data
                new_attr = edge_attr[2].values() + np.random.normal(0, .1, np.array(edge_attr[2].values()))
                edge_new_data[(edge_attr[0], edge_attr[1])] = {list(edge_attr[2].keys())[i]: new_attr[i]
                                                               for i in range(len(new_attr))
                                                               if list(edge_attr[2].keys())[i] != "label"}
            else:
                edge_new_data[(edge_attr[0], edge_attr[1])] = edge_attr[2]

        nx.set_node_attributes(graph, node_new_data)
        nx.set_edge_attributes(graph, edge_new_data)

        return graph

    @classmethod
    def _change_subgraph(cls, graph: GraphPlusDict, info: dict) -> (GraphPlusDict, int):
        """
        change subgraph that doesn't match to subgraph before changing
        :param graph:
        :param info:
        :return: new graph and change score
        """
        change_score = 0

        # change node label
        if 'label' in info['node'].keys():
            nodes_info = list(graph.nodes(data=True))
            node_new_data = dict()
            for node_attr in nodes_info:
                new_attr = node_attr[1]
                label, is_changed = cls._new_categorical_value(new_attr['label'], info['node']['label']['max'], info['node']['label']['min'])
                if is_changed:
                    change_score += 2
                    new_attr.update('label', label)
                node_new_data[node_attr[0]] = new_attr
            nx.set_node_attributes(graph, node_new_data)

        max_delete_edge = 0
        edges_info = list(graph.edges(data=True))
        edge_new_data = dict()
        for edge_attr in edges_info:
            # delete edge
            if random.random() < cls.DELETE_EDGE_P and max_delete_edge != 0 and \
                    cls._allow_delete_edge(graph, (edge_attr[0], edge_attr[1])):
                graph.remove_edge(edge_attr[0], edge_attr[1])
                max_delete_edge -= 1
                change_score += 1

            else:
                # change edge label
                if 'label' in info['edge'].keys():
                    new_attr = edge_attr[2]
                    label, is_changed = cls._new_categorical_value(new_attr['label'], info['edge']['label']['max'],
                                                                   info['edge']['label']['min'])
                    if is_changed:
                        change_score += 2
                        new_attr.update('label', label)
                    edge_new_data[(edge_attr[0], edge_attr[1])] = new_attr

        nx.set_edge_attributes(graph, edge_new_data)

        # add edge

        # add node

        return graph, change_score

    @classmethod
    def _new_categorical_value(cls, current_value: int, max_value: int, min_value: int) -> (int, bool):
        """
        return a new value for categorical attribute based on attribute range
        :param current_value:
        :param max_value:
        :param min_value:
        :return: new value, boolean to show value changed or not
        """
        if min_value == max_value:
            return min_value, False

        if random.random() < cls.CHANGE_LABEL_P:
            try_loop = 30
            while True and try_loop > 0:
                try_loop -= 1
                new_label = random.randint(min_value, max_value)
                if new_label != current_value:
                    return new_label, True
        return current_value, False

    @classmethod
    def _allow_delete_edge(cls, graph: GraphPlusDict, edge: tuple) -> bool:
        """
        can delete the edge (if all nodes of graph are connected, deleting edge must doesn't effect on it)
        :param graph:
        :param edge: (node_id_1, node_id_2)
        :return:
        """
        edge_list = list(graph.edges)
        nodes = set(list(graph.nodes))
        connected_nodes = {[edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]}
        if len(nodes - connected_nodes) > 0:
            return True

        edge_list.remove(edge)
        after_delete_connected_nodes = {[edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]}
        if len(nodes - after_delete_connected_nodes) > 0:
            return False

        return True

    @classmethod
    def _create_new_node(cls, graph: GraphPlusDict, info: dict) -> dict:
        """
        create new node based on graph
        :param graph:
        :param info:
        :return:
        """
        return dict()

    @classmethod
    def _create_new_edge(cls, graph: GraphPlusDict, info: dict, node_id=None) -> dict:
        """
        create new edge based on graph
        :param graph:
        :param info:
        :param node_id: can choose one of node_id
        :return:
        """
        
        return dict()

    @classmethod
    def _new_attributes_value(cls, attributes_info: dict) -> dict:
        """
        add noise to attributes based on attributes_info and return new values
        :param attributes_info:
        :return:
        """
        return dict()
