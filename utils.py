from torch_geometric.utils import to_dense_adj
import torch
from constants._connectivity import get_connectivity_weight
from constants._connectivity import CONNECTIVITY_TYPES
import numpy as np

def complex_format_fn(function_values, edge_index, complex_format='trispecific'):
    """
    Format the functional analysis results into a complex format.

    Args:
        function_values (torch.Tensor): Ehrlich results for all of the fabs. 
        edge_index (torch.Tensor): Edge index tensor representing the graph structure.
        complex_format (str): The type of complex format ('trispecific' or 'bispecific').

    Returns:
        torch.Tensor: A function value for the complex format.
    """
    # Get weights for this specific edge_index
    weights = get_connectivity_weight(complex_format, edge_index)
    
    # Convert edge_index to dense adjacency matrix
    A = to_dense_adj(edge_index)

    if complex_format == 'monospecific':
        # scaling factor to equalize variances
        k = np.sqrt((weights[0]**2 + weights[1]**2) / (weights[0] + weights[1])**2)
        weighted_function_values = 0.25 * k * (weights[0] + weights[1]) * function_values[0]
        return weighted_function_values
    
    elif complex_format == 'trispecific_example':
        if torch.equal(edge_index, CONNECTIVITY_TYPES['trispecific_example'][0]):
            y1 = function_values[2]
            y2 = function_values[0]
            y3 = function_values[1]
            synergy_term = 0.2
            co_expression_term = 0.1
            toxicity_penalty = 0.1
              
        elif torch.equal(edge_index, CONNECTIVITY_TYPES['trispecific_example'][1]):
            # fn_2 - fn_1 - fc - fn_0
            y1 = function_values[0] # switch y1 and y2
            y2 = function_values[2]
            y3 = function_values[1]
            synergy_term = 0.225
            co_expression_term = 0.1
            toxicity_penalty = 6

        else:
            raise ValueError(f"Edge index not found in trispecific_example connectivity types")
        return synergy_term * (y1*y2*y3) + co_expression_term * (y1*y2) - toxicity_penalty * (y2)
    else:
        weighted_function_values = weights * function_values

    # Compute the complex format value
        return (weighted_function_values + 
                weighted_function_values @ A + 
            weighted_function_values @ A @ A).sum()  / len(weighted_function_values)    

    
