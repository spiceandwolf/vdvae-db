import csv
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchquad import MonteCarlo, set_up_backend, VEGAS
import data_tabular 

'''
    Utility functions for sql.
'''
OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


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
    # ops = rng.choice(['<=', '>='], size=num_filters)
    # ops = rng.choice(['='], size=num_filters)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def Card(table_data, columns, operators, vals):
    assert len(columns) == len(operators) == len(vals)

    bools = None
    for c, o, v in zip(columns, operators, vals):
        inds = OPS[o](table_data[c.name], v)

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    return c


def FillInUnqueriedColumns(table, columns, operators, vals):
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


def Query(table, columns, operators, vals, n_samples):
        columns, operators, vals = FillInUnqueriedColumns(table, columns, operators, vals)

        all_samples = []
        for column, op, val in zip(columns, operators, vals):
            if op is not None:
                valid = OPS[op](column['all_distinct_values'], val)
            else:
                valid = [True] * len(column['all_distinct_values'])
            
            selected_idx = [i for i, selected in enumerate(valid) if selected]
            samples = np.random.choice(selected_idx, n_samples)
            # print([column_info['all_distinct_values'][sample] for sample in samples][0:10])
            all_samples.append([column['all_distinct_values'][sample] for sample in samples])
        all_samples = np.asarray(all_samples).T

        return all_samples


'''
    Utility functions for integrate.
'''
def test_integrate(attrs, min = 0, max = 1, alias2table=None):
    left_bounds = {}
    right_bounds = {}
    
    for attr in attrs:
        col_name = attr
        if(len(attr.split('.')) == 2):
            if alias2table is None:
                col_name = alias2table[attr.split('.')[0]] + f".{attr.split('.')[1]}"
                
        left_bounds[col_name] = min
        right_bounds[col_name] = max
                
    integration_domain = []
    for attr in attrs:
        integration_domain.append([left_bounds[attr], right_bounds[attr]])
                
    return integration_domain


def make_points(table_stats, predicates, bias, noise_type=None, normalize=None):
    attrs, name_to_index, right_bounds, left_bounds = table_stats
    
    import copy
    maxs = copy.deepcopy(right_bounds)
    mins = copy.deepcopy(left_bounds)
    
    for predicate in predicates:
        if len(predicate) == 3:
            
            column = predicate[0].name # 适用于imdb的
            operator = predicate[1]
            val = float(predicate[2])
            
            if noise_type == 'uniform':
                if operator == '=':
                    left_bounds[column] = val
                    right_bounds[column] = val + 2 * bias[name_to_index[column]]
                
                elif operator == '<=':
                    left_bounds[column] = mins[column] 
                    right_bounds[column] = val + 2 * bias[name_to_index[column]] 
                    
                elif operator  == ">=":
                    left_bounds[column] = val 
                    right_bounds[column] = maxs[column] 
            else:
                if operator == '=':
                    left_bounds[column] = val - bias[name_to_index[column]] 
                    right_bounds[column] = val + bias[name_to_index[column]] 
                    
                elif operator == '<=':
                    left_bounds[column] = mins[column] 
                    right_bounds[column] = val + bias[name_to_index[column]] 
                    
                elif operator  == ">=":
                    left_bounds[column] = val - bias[name_to_index[column]] 
                    right_bounds[column] = maxs[column]
    
    # print(f'left_bounds {left_bounds} right_bounds {right_bounds}')
                        
    integration_domain = []
    
    for attr in attrs:
        attr = attr.name
        assert left_bounds[attr] < right_bounds[attr], f'predicates {predicates} attr {attr} left_bounds {left_bounds[attr]} right_bounds {right_bounds[attr]}'
        if normalize == 'minmax':
            integration_domain.append([(left_bounds[attr] - mins[attr]) / (maxs[attr] - mins[attr]), (right_bounds[attr] - mins[attr]) / (maxs[attr] - mins[attr])])
            
        elif normalize  == 'normalize':
            loc = 0.5 * (maxs[attr] + mins[attr])
            integration_domain.append([(left_bounds[attr] - loc) / (0.5 * maxs[attr] - 0.5 * mins[attr]), (right_bounds[attr] - loc) / (0.5 * maxs[attr] - 0.5 * mins[attr])])
            
        # elif normalize == 'integer':
        #     scale = 0.5 / shift
        #     out = out * scale
                
    return integration_domain


def estimate_probabilities(pdf, integration_domain, dim, isdiscrete = False):
    integration_domain = torch.Tensor(integration_domain).cuda()
    
    if isdiscrete:
        pass
        # x = integration_domain.reshape(-1, 1, integration_domain.shape[1])
        # prob = torch.exp(model.elbo(x)).sum()
    
    else:
        set_up_backend("torch", data_type="float32")
        def multivariate_normal(x):
            x = x.reshape(-1, 1, x.shape[1])
            # print(f'x:{x.shape}')
            with torch.no_grad():
                # elbo = model.modules.elbo(x)
                # print(f'x:{x.shape}')
                elbo = pdf(x)
                # print(f'elbo:{elbo.shape}')
                prob_list = torch.exp(elbo)
                return prob_list
            
        integrater = MonteCarlo()
        # integrater = VEGAS()
        prob = integrater.integrate(
            multivariate_normal,
            dim=dim,
            N=1000,
            integration_domain=integration_domain,
            backend="torch",
            )   
        
    return prob


'''
    Utility functions for data.
'''
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


def probs2contours(probs, levels):
    """
    Takes an array of probabilities and produces an array of contours at specified percentile levels
    :param probs: probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    :param levels: percentile levels. have to be in [0.0, 1.0]
    :return: array of same shape as probs with percentile labels
    """

    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf_marginals(pdf, lims, gt=None, levels=(0.68, 0.95)):
    """
    Plots marginals of a pdf, for each variable and pair of variables.
    """

    if pdf.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(lims[0], lims[1], 200)

        pp = pdf.eval(xx[:, np.newaxis], log=False)
        ax.plot(xx, pp)
        ax.set_xlim(lims)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        fig, ax = plt.subplots(pdf.ndim, pdf.ndim)

        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf.ndim, 1]) if lims.ndim == 1 else lims

        for i in range(pdf.ndim):
            for j in range(pdf.ndim):

                if i == j:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 500)
                    pp = pdf.eval(xx, ii=[i], log=False)
                    ax[i, j].plot(xx, pp)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 200)
                    yy = np.linspace(lims[j ,0], lims[j, 1], 200)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[i, j], log=False)
                    pp = pp.reshape(list(X.shape))
                    ax[i, j].contour(X, Y, probs2contours(pp, levels), levels)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax


def plot_hist_marginals(data, lims=None, gt=None):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, normed=True)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if lims is not None: ax.set_xlim(lims)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        n_dim = data.shape[1]
        fig, ax = plt.subplots(n_dim, n_dim)
        ax = np.array([[ax]]) if n_dim == 1 else ax

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):

                if i == j:
                    ax[i, j].hist(data[:, i], n_bins, normed=True)
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if lims is not None: ax[i, j].set_xlim(lims[i])
                    if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                        ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax



'''
    Utility functions for metrics.
'''
def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


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
    
    card = Card(table_data, cols, ops, vals)
    print(card)
