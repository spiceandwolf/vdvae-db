import csv
import os
import numpy as np
import pandas as pd
import torch
from torchquad import MonteCarlo, set_up_backend, VEGAS


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}

def make_points(attrs, predicates, statistics, bias, alias2table=None):
    left_bounds = {}
    right_bounds = {}
    
    for attr in attrs:
        col_name = attr
        if(len(attr.split('.')) == 2):
            if alias2table is None:
                col_name = alias2table[attr.split('.')[0]] + f".{attr.split('.')[1]}"
                
        left_bounds[col_name] = 0
        right_bounds[col_name] = 1
    
    for predicate in predicates:
        if len(predicate) == 3:
            
            column = predicate[0] # 适用于imdb的
            operator = predicate[1]
            val = float(predicate[2])
                
            if operator == '=':
                left_bounds[column] = (val - bias[column] - statistics[column]['min']) / (statistics[column]['max'] - statistics[column]['min'])
                right_bounds[column] = (val + bias[column] - statistics[column]['min']) / (statistics[column]['max'] - statistics[column]['min'])
                
            elif operator == '<=':
                right_bounds[column] = (val - statistics[column]['min']) / (statistics[column]['max'] - statistics[column]['min'])
                
            elif operator  == ">=":
                left_bounds[column] = (val - statistics[column]['min']) / (statistics[column]['max'] - statistics[column]['min'])
                
    integration_domain = []
    for attr in attrs:
        integration_domain.append([left_bounds[attr], right_bounds[attr]])
                
    return integration_domain

def make_point_raw(attrs, predicates, statistics, bias, alias2table=None):
    left_bounds = {}
    right_bounds = {}
    
    for attr in attrs:
        col_name = attr
        if(len(attr.split('.')) == 2):
            if alias2table is None:
                col_name = alias2table[attr.split('.')[0]] + f".{attr.split('.')[1]}"
                
        left_bounds[col_name] = statistics[col_name]['min']
        right_bounds[col_name] = statistics[col_name]['max']
    
    for predicate in predicates:
        if len(predicate) == 3:
            
            column = predicate[0] # 适用于imdb的
            operator = predicate[1]
            val = float(predicate[2])
                
            if operator == '=':
                left_bounds[column] = val - bias[column] 
                right_bounds[column] = val + bias[column] 
                
            elif operator == '<=':
                right_bounds[column] = val
                
            elif operator  == ">=":
                left_bounds[column] = val
                
    integration_domain = []
    for attr in attrs:
        integration_domain.append([left_bounds[attr], right_bounds[attr]])
                
    return integration_domain


def estimate_probabilities(model, integration_domain, dim):
    integration_domain = torch.Tensor(integration_domain).cuda()
    
    set_up_backend("torch", data_type="float32")
    def multivariate_normal(x):
        x = x.reshape(-1, 1, x.shape[1])
        # print(f'x:{x}')
        with torch.no_grad():
            nelbo = model.module.nelbo(x)
            # nelbo = model.nelbo(x)
            prob_list = torch.exp(nelbo)
            return prob_list
        
    mc = MonteCarlo()
    prob = mc.integrate(
        multivariate_normal,
        dim=dim,
        N=10000,
        integration_domain=integration_domain,
        backend="torch",
        )   
    return prob


def load_data(file_name):
    joins = []
    predicates = []
    tables = []
    label = []

    # Load queries
    with open(file_name, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")
    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, label


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
        
def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def get_col_statistics(table_data, statistics_file, alias2table=None):

    names = []
    cards = []
    distinct_nums = []
    mins = []
    maxs = []

    for attribute in table_data.columns:
        col = attribute.split('.')[-1]
        if alias2table is not None:
            name = + f"{alias2table[attribute.split('.')[0]]}.{col}"
        else:
            name = attribute
        names.append(name)
       
        col_materialize = table_data[attribute]
        maxs.append(col_materialize.max())
        mins.append(col_materialize.min())
        cards.append(len(col_materialize))
        distinct_nums.append(len(col_materialize.unique()))
        
    statistics = pd.DataFrame(
        data={'name': names, 'min': mins, 'max': maxs, 'cardinality': cards, 'num_unique_values': distinct_nums})
    if os.path.exists(statistics_file):
        statistics.to_csv(statistics_file, index=False, mode='a', header=None)
    else:
        statistics.to_csv(statistics_file, index=False)
    return statistics.to_dict('list')


def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    num_filters = rng.randint(1, 7)
    # num_filters=1
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.iloc[rng.randint(0, table.shape[0])]
    vals = s.values

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)

    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    # ops = rng.choice(['='])

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def Card(table, columns, operators, vals):
    assert len(columns) == len(operators) == len(vals)

    bools = None
    for c, o, v in zip(columns, operators, vals):
        inds = OPS[o](table[c], v)

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    return c

if __name__ == "__main__":
    data_root = '~/QOlab/dataset/'
    table_data = pd.read_csv(os.path.join(data_root, 'household_power_consumption.txt'), delimiter=';', 
                               usecols=[2,3,4,5,6,7,8], na_values=[' ', '?'])
    table_data = table_data.dropna(axis=0, how='any')
    # get_col_statistics(table_data, './power/statistics.csv')
    result = table_data[table_data['Global_intensity']==1.4]
    print(result)
    rng = np.random.RandomState(1234)
    cols, ops, vals = GenerateQuery(table_data.columns, rng, table_data)
    print(cols, ops, vals)
    
    card = card(table_data, cols, ops, vals)
    print(card)
