import numpy as np
from tqdm import tqdm
from question_loader import *
from utils import *
from collections import deque
from LLM import *
from itertools import permutations

##### SETTINGS #####
cache_dir = '/tmp'
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
data_outpath = './data/experiment'
######################

def topological_sort(adjacency_matrix):
    # Number of nodes in the graph
    num_nodes = len(adjacency_matrix)
    
    # Calculate in-degrees of all nodes
    in_degree = [0] * num_nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                in_degree[j] += 1
    
    # Initialize a queue with all nodes having in-degree of 0
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    
    topological_order = []
    
    while queue:
        node = queue.popleft()
        topological_order.append(node)
        
        # Reduce the in-degree of all neighbors by 1
        for neighbor in range(num_nodes):
            if adjacency_matrix[node][neighbor] != 0:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    # If the topological order includes all nodes, return it
    if len(topological_order) == num_nodes:
        return topological_order
    else:
        # There is a cycle, return None
        return None

def count_violations(order, adj_matrix):
    violations = 0
    n = len(order)
    for i in range(n):
        for j in range(0, i):
            violations += adj_matrix[order[j], order[i]]    
    return violations

def minimum_violations_topo_sort(adj_matrix):
    # n = len(adj_matrix)
    # nodes = list(range(n))
    
    # min_violations = float('inf')
    # best_order = None
    
    # for order in permutations(nodes):
    #     violations = count_violations(order, adj_matrix)
    #     if violations < min_violations:
    #         min_violations = violations
    #         best_order = order
    
    # # print(best_order, min_violations)
    # return (best_order, min_violations)
    
    res = (-1e9, -1)
    for i in range(len(adj_matrix)):
        cur = 0
        for j in range(len(adj_matrix)):
            cur += adj_matrix[i, j]
            cur -= adj_matrix[j, i]
            
        res = max(res, (cur, i))
        
    print(res)
    return (res[1], res[0])
            
n_samples = 100

res = []
correct_count = 0

with tqdm(total=n_samples) as pbar:
    for row in range(n_samples):
        row_options = getOptionsArr(row)
        M = len(row_options)
        adj_matrix = np.zeros((M, M))
        
        get_prompt = get_compare_query_func(row)
        
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                    
                cur = get_prompt(i, j)
                
                # print('=' * 50)
                # print(i, j)
                # print('=' * 50)
                # print(cur)
                # print('=' * 50)
                
                response, probs = get_next_token_fast(cur)
                
                # print("BRUH", cur, response, probs)
                # print("RESPONSE", i, j, response[0])
                print(response, probs)
                if response[0] == 'A':
                    adj_matrix[i, j] += probs[0] # i preferred over j.... Try prob
                elif response[0] == 'B':
                    adj_matrix[j, i] += probs[0] # j perferred over i
                # else:
                #     print("None")

        # print(adj_matrix)
        print_graph_from_adj_matrix(adj_matrix)
        # order = topological_sort(adj_matrix)
        # violations = 0
        
        # if order is None:
        order, violations = minimum_violations_topo_sort(adj_matrix)
        
        # model_ans = row_options[order[0]][0]
        # model_ans = row_options[order[-1]][0] ## CHANGE THISS
        print(order, violations)
        model_ans = row_options[int(order)][0] ## CHANGE THISS
        cor_ans = get_correct_answer(row)
        
        print(model_ans, cor_ans, order, row_options)
        assert cor_ans in possible_outputs

        # print(order, row_options, cor_ans)

        is_correct = model_ans == cor_ans
        if is_correct:
            correct_count += 1

        pbar.set_postfix({'Correct %': f'{(correct_count / (row + 1)) * 100:.2f}%'})
        pbar.update(1)

        # print(row, is_correct, entropy, model_ans, cor_ans, probs[0])
        res.append({
                "row": row,
                "violations": violations,
                "is_correct": is_correct,
                "model_prob":probs[0],
                "model_response":response[0]
            })

dump_data(res, data_outpath)
