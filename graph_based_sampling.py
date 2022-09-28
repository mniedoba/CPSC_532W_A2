from graphlib import TopologicalSorter

# Project imports
from evaluation_based_sampling import Expression, eval_expression

class Graph:
    def __init__(self, graph_json):
        self.json = graph_json
        # Convert to node: predecessor format from node: successor format.
        node_preds = {node: set() for node in self.json[1]['V']}
        for node, successors in self.json[1]['A'].items():
            for successor in successors:
                node_preds[successor].add(node)

        # Sort the nodes in topological order and save off the ordered list of nodes.
        self.topological_sorter = TopologicalSorter(node_preds)
        self.ordered_nodes = [node for node in self.topological_sorter.static_order()]

        # Link functions are mappings from nodes to expressions.
        self.link_functions = {k: Expression(v) for k, v in self.json[1]['P'].items()}

        # Procedures don't seem to be used in the syntax at the moment?? Not really sure what is up with that.
        self.procedures = self.json[0] #Todo.

        # The return of the program is just an expression.
        self.return_expression = Expression(self.json[-1])


def evaluate_graph(graph, verbose=False):

    node_vals = {}
    sigma =[]

    # Loop through the topological ordering of nodes, updating the environment as we go.
    # TODO: Do I need to think about scope here?
    for node in graph.ordered_nodes:
        node_vals[node], sigma = eval_expression(graph.link_functions[node], sigma, node_vals, graph.procedures)

    # Evaluate the return expression with the sampled values.
    r_val, sigma = eval_expression(graph.return_expression, sigma, node_vals, graph.procedures)
    return r_val, sigma, node_vals