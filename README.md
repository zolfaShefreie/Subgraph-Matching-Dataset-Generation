# Subgraph Matching Dataset Generation
Create dataset from graph datasets for subgraph matching learning<br/>

## Graph dataset
Choosen datasets for converting to new dataset:
-   AIDS
-   BZR
-   Cuneiform
-   IMDB-MULTI <br/>
these dataset store on **graph_datasets** folder and you can download more graph dataset from [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
## Subgraph matching dataset
each subgraph matching dataset is sore as file that these files is stores at **subgraph_matching_dataset** folder.<br/>
every row of dataset file have below format:<br/>
{'source_graph': {'nodes': {node_id: node_feature_dictionary, }, 'edge': {(node_id_1, node_id_2): edge_feature_dict, ...}},<br/>
  'query_graph': {'nodes': {node_id: node_feature_dictionary, }, 'edge': {(node_id_1, node_id_2): edge_feature_dict, ...}},<br/>
  'label': 1}<br/>
-   **source_graph** is a one of graph in graph dataset
-   **query_graph** is a matched or unmatched subgraph of **source_graph**
-   **label** can be 0 or 1 to show **query_graph** is subgraph of **source_graph** or not
### How the query_match is created?
all of below changes applied with some probabilities. changes:
1. get the random subgraph from source_graph
2. add noise to node and edge feature and delete one edge based on conditions
3. change the labels of node and edges, delete edges, add nodes, and add edges
