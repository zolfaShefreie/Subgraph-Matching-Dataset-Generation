import networkx as nx
from os import listdir
from os.path import isfile, join
import random
import itertools
import numpy as np
import gzip
import json
import os


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

    def __init__(self, source_graph: GraphPlusDict, query_graph: GraphPlusDict, label):
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

    # def to_json_encode(self):
    #     """
    #     encode json of element
    #     :return:
    #     """
    #     json_str = json.dumps(self.to_dict()) + "\n"
    #     return json_str.encode('utf-8')

    def __str__(self):
        """
        str of element
        :return:
        """
        return str(self.to_dict()) + "\n"


class SubGraphDatasetGenerator:
    """
    this class generate dataset for subgraph matching problem.
    stages for create dataset:
        1. read main graph dataset
        2. create graphs one by one
        3. create subgraphs that match or doesn't match the graph
        4. save the elements of new dataset
    """

    CATEGORICAL_UNIQUE_NUMBER = 15
    NODE_NOISE_THRESHOLD = 0.2
    EDGE_NOISE_THRESHOLD = 0.3
    ATTRIBUTE_NOISE_THRESHOLD = 0.7
    DELETE_EDGE_P = 0.6
    CHANGE_LABEL_P = 0.4
    ADD_EDGE = 0.3
    ADD_NODE = 0.2
    ADD_NODE_THRESHOLD = 1
    ADD_EDGE_THRESHOLD = 5

    @classmethod
    def generate(cls, dataset_dir: str, output_dir):
        files = cls._get_dataset_open_files(dataset_dir)
        info_dict = cls._scan_to_get_info(files)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = gzip.open(f"{output_dir}/{dataset_dir.split('/')[-1]}.txt.gz", 'wt')

        graph = cls._create_graph_from_file(files)
        while nx.number_of_nodes(graph) != 0:
            cls._make_save_elements_of_graph(graph, info_dict, output_file)
            graph = cls._create_graph_from_file(files)
        
        output_file.close()

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
        info = {'node': {}, 'edge': {}}
        for file_name in ['node_labels', 'node_attributes', 'edge_labels', 'edge_attributes']:
            if file_name in files.keys():

                base_feature_name = "label" if file_name.split("_")[-1] == "labels" else "attr"
                edge_or_node = file_name.split("_")[0]

                feature_dict = dict()
                lines = files[file_name]['file'].readlines()

                # spilt features and save if to a dictionary like {feature_1: {value_1, value_2, ..}}
                for line in lines:
                    if line != "\n":
                        elements = line.split('\n')[0].split(', ')
                        for i, element in enumerate(elements):

                            value_set = feature_dict.get(f"{base_feature_name}_{i}", set())
                            if base_feature_name == "label":
                                value_set.add(int(element))
                            else:
                                value_set.add(float(element))
                            feature_dict[f"{base_feature_name}_{i}"] = value_set

                # summarize info for one file
                info[edge_or_node].update({key: {"max": max(value),
                                                 "min": min(value),
                                                 "kind": "category" if (base_feature_name == 'label' or
                                                                        len(value) <= cls.CATEGORICAL_UNIQUE_NUMBER)
                                                 else "number",
                                                 "values": list(value) if (base_feature_name == 'label' or
                                                                                len(value) <= cls.CATEGORICAL_UNIQUE_NUMBER)
                                                 else list()}
                                           for key, value in feature_dict.items()})

                # change position file to read from first byte
                files[file_name]['file'].seek(0)

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
            if line == "":
                return graph

            if last_graph_id != -1 and last_graph_id != line.split('\n')[0]:
                files['graph_indicator']['file'].seek(files['graph_indicator']['position'], 0)
                break

            last_graph_id = line.split('\n')[0]
            files['graph_indicator']['position'] = files['graph_indicator']['file'].tell()
            files['graph_indicator']['line'] = files['graph_indicator']['line'] + 1

            node_atrrs = dict()

            if 'node_labels' in files.keys():
                node_labels = files['node_labels']['file'].readline().split('\n')[0].split(', ')
                node_atrrs.update({f'label_{index}': int(label) for index, label in enumerate(node_labels)})
                files['node_labels']['position'] = files['node_labels']['file'].tell()
                files['node_labels']['line'] = files['node_labels']['line'] + 1

            if 'node_attributes' in files.keys():
                node_attributes = files['node_attributes']['file'].readline().split('\n')[0].split(', ')
                node_atrrs.update({f'attr_{index}': float(attr) for index, attr in enumerate(node_attributes)})
                files['node_attributes']['position'] = files['node_attributes']['file'].tell()
                files['node_attributes']['line'] = files['node_attributes']['line'] + 1

            graph.add_node(files['graph_indicator']['line'], **node_atrrs)

        # adding edges and their attributes
        graph_nodes = set(list(graph.nodes))
        while True:
            line = files['A']['file'].readline()
            if line == "":
                return graph

            node_ids = [int(node_id) for node_id in line.split('\n')[0].split(', ')]
            if len(set(node_ids) - graph_nodes) > 0:
                files['A']['file'].seek(files['A']['position'], 0)
                break
            files['A']['position'] = files['A']['file'].tell()
            files['A']['line'] = files['A']['line'] + 1

            edge_attrs = dict()
            if 'edge_labels' in files.keys():
                edge_labels = files['edge_labels']['file'].readline().split('\n')[0].split(', ')
                edge_attrs.update({f'label_{index}': int(label) for index, label in enumerate(edge_labels)})
                files['edge_labels']['position'] = files['edge_labels']['file'].tell()
                files['edge_labels']['line'] = files['edge_labels']['line'] + 1

            if 'edge_attributes' in files.keys():
                edge_attributes = files['edge_attributes']['file'].readline().split('\n')[0].split(', ')
                edge_attrs.update({f'attr_{index}': float(attr) for index, attr in enumerate(edge_attributes)})
                files['edge_attributes']['position'] = files['edge_attributes']['file'].tell()
                files['edge_attributes']['line'] = files['edge_attributes']['line'] + 1

            graph.add_edge(node_ids[0], node_ids[1], **edge_attrs)

        return graph

    @classmethod
    def _make_save_elements_of_graph(cls, graph: GraphPlusDict, info: dict, file):
        """
        make pairs and save it
        :param graph:
        :param info:
        :param path:
        :return: nothing
        """
        if nx.number_of_nodes(graph) <= 4:
            return

        for i in range(5):
            subgraph = cls._add_noise_to_graph(cls._get_random_subgraph(graph))
            element = SubGraphMatchingElement(source_graph=graph, query_graph=subgraph, label=1)
            file.write(str(element))

        for i in range(5):
            subgraph = cls._add_noise_to_graph(cls._get_random_subgraph(graph))
            subgraph, score = cls._change_subgraph(subgraph, info)
            element = SubGraphMatchingElement(source_graph=graph, query_graph=subgraph, label=0 if score > 1 else 1)
            file.write(str(element))

    @classmethod
    def _get_random_subgraph(cls, graph: GraphPlusDict) -> GraphPlusDict:
        """
        return a random subgraph
        :param graph:
        :return:
        """
        graph_nodes = set(list(graph.nodes))
        random_len_subgraph = random.randint(int(len(graph_nodes)/2)+1, len(graph_nodes) - cls.ADD_NODE_THRESHOLD)
        subgraph_nodes = list()
        for i in range(random_len_subgraph):
            choice = random.choice(list(graph_nodes))
            subgraph_nodes.append(choice)
            graph_nodes.remove(choice)
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
        graph = GraphPlusDict(graph)
        max_delete_edge = 1 if nx.number_of_nodes(graph) >= 3 else 0

        # add noise to node
        nodes_info = list(graph.nodes(data=True))
        node_new_data = dict()
        for node_attr in nodes_info:
            if random.random() < cls.NODE_NOISE_THRESHOLD:
                new_attr = np.array(list(node_attr[1].values())) + np.random.normal(0, .1, np.array(list(node_attr[1].values())).shape)
                node_new_data[node_attr[0]] = {list(node_attr[1].keys())[i]: new_attr[i] for i in range(len(new_attr))
                                               if "label" not in list(node_attr[1].keys())[i]}

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
                new_attr = np.array(list(edge_attr[2].values())) + np.random.normal(0, .1, np.array(list(edge_attr[2].values())).shape)
                edge_new_data[(edge_attr[0], edge_attr[1])] = {list(edge_attr[2].keys())[i]: new_attr[i]
                                                               for i in range(len(new_attr))
                                                               if "label" not in list(edge_attr[2].keys())[i]}

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
        graph = GraphPlusDict(graph)

        # change node label
        if 'label_0' in info['node'].keys():
            nodes_info = list(graph.nodes(data=True))
            node_new_data = dict()
            for node_attr in nodes_info:
                new_attr = node_attr[1]
                for feature_name in new_attr.keys():
                    if "label" in feature_name:

                        label, is_changed = cls._new_categorical_value(new_attr[feature_name],
                                                                       info['node'][feature_name]['values'],
                                                                       True)
                        if is_changed:
                            change_score += 2
                            new_attr.update({feature_name: label})

                node_new_data[node_attr[0]] = new_attr
            nx.set_node_attributes(graph, node_new_data)

        max_delete_edge = random.randint(0, int(nx.number_of_edges(graph)/4))
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
                if 'label_0' in info['edge'].keys():
                    new_attr = edge_attr[2]
                    for feature_name in new_attr:
                        if "label" in feature_name:
                            label, is_changed = cls._new_categorical_value(new_attr[feature_name],
                                                                           info['edge'][feature_name]['values'],
                                                                           True)
                            if is_changed:
                                change_score += 2
                                new_attr.update({feature_name: label})
                    edge_new_data[(edge_attr[0], edge_attr[1])] = new_attr

        nx.set_edge_attributes(graph, edge_new_data)

        # add edge
        for i in range(random.randint(0, cls.ADD_EDGE_THRESHOLD)):
            new_edge = cls._create_new_edge(graph, info)
            if not new_edge:
                break
            graph.add_edge(new_edge['edge'][0], new_edge['edge'][0], **new_edge['attr'])
            change_score += 2

        # add node
        for i in range(random.randint(0, cls.ADD_NODE_THRESHOLD)):
            new_node = cls._create_new_node(graph, info)
            graph.add_node(new_node['node'], **new_node['attr'])
            change_score += 2

        return graph, change_score

    @classmethod
    def _new_categorical_value(cls, current_value, values: list, check_threshold=False) -> (int, bool):
        """
        return a new value for categorical attribute based on attribute range
        :param current_value:
        :param values: unique values of this feature
        :return: new value, boolean to show value changed or not
        """
        if len(values) == 1:
            return values[0], False if current_value else True

        if ((random.random() < cls.CHANGE_LABEL_P and check_threshold) or not check_threshold) and len(values) > 1:
            try_loop = 30
            while True and try_loop > 0:
                try_loop -= 1
                new_label = random.choice(values)
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
        connected_nodes = set([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list])
        if len(nodes - connected_nodes) > 0:
            return True

        edge_list.remove(edge)
        after_delete_connected_nodes = set([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list])
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
        new_id = max(list(graph.nodes)) + 1
        return {'node': new_id, 'attr': cls._new_attributes_value(info['node'])}

    @classmethod
    def _create_new_edge(cls, graph: GraphPlusDict, info: dict, node_id=None) -> dict:
        """
        create new edge based on graph
        :param graph:
        :param info:
        :param node_id: can choose one of node_id
        :return:
        """
        if node_id:
            not_connected_nodes = list(set(list(graph.nodes)) - set([x for x in nx.all_neighbors(graph, node_id)]))
            if len(not_connected_nodes) == 0:
                return dict()
            other_node_id = random.choice(not_connected_nodes)
            edge = (node_id, other_node_id)
        else:
            non_edges = [x for x in nx.non_edges(graph)]
            if len(non_edges) == 0:
                return dict()
            edge = random.choice(non_edges)

        new_attr = cls._new_attributes_value(info['edge'])
        return {'edge': edge, 'attr': new_attr}

    @classmethod
    def _new_attributes_value(cls, attributes_info: dict) -> dict:
        """
        add noise to attributes based on attributes_info and return new values
        :param attributes_info:
        :return:
        """
        new_attributes = dict()
        for key, value in attributes_info.items():
            if value['kind'] == 'category':
                new_attributes['key'], _ = cls._new_categorical_value(None, value['values'])
            else:
                new_attributes['key'] = random.uniform(value['min'], value['max'])
        
        return new_attributes


if __name__ == "__main__":
    graph_dataset_dir = "./graph_datasets"
    dirs = {dir_name: f"{graph_dataset_dir}/{dir_name}" for dir_name in os.listdir(graph_dataset_dir)
            if os.path.isdir(f"{graph_dataset_dir}/{dir_name}")}
    for key, value in dirs.items():
        SubGraphDatasetGenerator.generate(value, "./subgraph_matching_dataset")
        print(key, " is completed")
