import copy
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def set_up_data(H):

    untranspose = False
    
    if H.dataset == 'power':
        
        original_data = power(H)
        
        H.image_size = sum(original_data.cols_size) if H.discrete else 7
        H.image_channels = 1
        H.distortions = [1 / i.DistributionSize() for i in original_data.columns]
        
        H.encoding_lists = []
        if H.encoding_modes is not None:
            for ss in H.encoding_modes.split(','):
                H.encoding_lists.append(ss)
        
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    shift = torch.tensor(original_data.bias).cuda().view(1, 1, H.image_size)
    data_max_ = torch.tensor(original_data.maxs).cuda().view(1, 1, H.image_size)
    data_min_ = torch.tensor(original_data.mins).cuda().view(1, 1, H.image_size)
    H.distortions = torch.tensor([1 / i.DistributionSize() for i in original_data.columns]).cuda().view(1, 1, H.image_size)

    noise_type = H.noise_type
    normalize = H.normalize
    data_max = data_max_
    data_min = data_min_
    
    if noise_type == 'uniform':
        '''
        bin = 2 * shift
        [a, a + bin)
        domian : [data_min_, data_max_ + bin)
        '''
        data_max = data_max_ + 2 * shift
        data_min = data_min_
        
    elif noise_type == 'gaussian':
        '''
        (a - shift, a + shift)
        bin = 2 * shift
        domian : (data_min_ - bin, data_max_ + bin)
        '''
        data_max = data_max_ + 2 * shift 
        data_min = data_min_ - 2 * shift
        
    H.sigma = (shift / (data_max - data_min)).float()
    H.shift = (H.sigma).float()
    print(f'shift {H.shift}')

    split_index = int(original_data.data.shape[0] * 0.1)
    original_data = TableDataset(original_data) if H.discrete else original_data
    teX = original_data.data[:split_index].values.reshape(-1, 1, H.image_size)
    vaX = teX
    trX = original_data.data[split_index:].values.reshape(-1, 1, H.image_size)
    
    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX        
        
    train_data = TensorDataset(torch.as_tensor(trX))
    valid_data = TensorDataset(torch.as_tensor(eval_dataset))

    def preprocess_func(x):
        nonlocal shift
        nonlocal data_max
        nonlocal data_min
        nonlocal noise_type
        nonlocal normalize
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 1)
        inp = x[0].cuda(non_blocking=True)
        out = inp.clone()
        # print(f'noise {noise}')
        
        if H.discrete:
            out = out.long()
            data = [None] * len(original_data.cols_size)
            const_one = torch.ones([], dtype=torch.long, device=data.device)
            dist_start = 0
            output = [None] * len(original_data.cols_size)
            for i, col_dom_size in enumerate(original_data.cols_size): 
                
                if H.encoding_lists[i] == 'binary':
                    data[i] = const_one << torch.arange(col_dom_size, device=data.device)
                    col_data = out.narrow(1, i, 1)
                    binaries = (col_data & data[i]) > 0
                    output[i] = binaries
                
                elif H.encoding_lists[i] == 'onehot':
                    onehot = torch.zeros(bs, col_dom_size, device=data.device)
                    onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                    output[i] = onehot
                
            out = torch.cat(output, 1)
        
        else:    
            scale = 0
            
            if noise_type == 'uniform':
                scale = torch.empty_like(inp).uniform_(0, 1) * 2

            elif noise_type == 'gaussian':
                scale = torch.empty_like(inp).normal_(0, 1 / 3) 
                
            elif noise_type == 'None':
                scale = torch.zeros_like(inp)
                
            out = out + scale * shift
                
            # print(f'out {out[0]}')
            if normalize == 'minmax':
                out = (out - data_min) / (data_max - data_min)
                
            elif normalize  == 'normalize':
                loc = 0.5 * (data_max + data_min)
                out = (out - loc) / (0.5 * data_max - 0.5 * data_min)
                
            elif normalize == 'integer':
                scale = 0.5 / shift
                out = out * scale
        
        return inp.float(), out.float()

    return H, train_data, valid_data, preprocess_func, original_data


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def power(H):
    csv_file = os.path.join(H.data_root, 'household_power_consumption.txt')
    cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    trX = CsvTable(H, 'power', csv_file, cols, sep=';', na_values=[' ', '?'], header=0, dtype=np.float64)      
    trX.data = trX.data.dropna(axis=0, how='any')
    trX.data = trX.data.sample(frac=1).reset_index(drop=True)
    
    return trX


# Naru. https://github.com/naru-project/naru.git

class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(self, name, distribution_size=None, pg_name=None):
        self.name = name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.distribution_size = distribution_size

        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def ValToBin(self, val):
        if isinstance(self.all_distinct_values, list):
            return self.all_distinct_values.index(val)
        inds = np.where(self.all_distinct_values == val)
        assert len(inds[0]) > 0, val

        return inds[0][0]

    def SetDistribution(self, distinct_values):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        contains_nan = np.any(is_nan)
        dv_no_nan = distinct_values[~is_nan]
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(dv_no_nan))
        # if contains_nan and np.issubdtype(distinct_values.dtype, np.datetime64):
        #     vs = np.insert(vs, 0, np.datetime64('NaT'))
        # elif contains_nan:
        #     vs = np.insert(vs, 0, np.nan)
        if self.distribution_size is not None:
            assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        self.distribution_size = len(vs)
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data)
        return self

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        self.cardinality = self._validate_cardinality(columns)
        self.columns = columns

        self.val_to_bin_funcs = [c.ValToBin for c in columns]
        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}

        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name
            
        self

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index
        return self.name_to_index[name]


class CsvTable(Table):
    """Wraps a CSV file or pd.DataFrame as a Table."""

    def __init__(self,
                 H,
                 name,
                 filename_or_df,
                 cols,
                 type_casts={},
                 pg_name=None,
                 pg_cols=None,
                 **kwargs):
        """Accepts the same arguments as pd.read_csv().

        Args:
            H: hps list.
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
            cols: list of column names to load; can be a subset of all columns.
            type_casts: optional, dict mapping column name to the desired numpy
              datatype.
            pg_name: optional str, a convenient field for specifying what name
              this table holds in a Postgres database.
            pg_name: optional list of str, a convenient field for specifying
              what names this table's columns hold in a Postgres database.
            **kwargs: keyword arguments that will be pass to pd.read_csv().
        """
        self.name = name
        self.pg_name = pg_name

        if isinstance(filename_or_df, str):
            self.data = self._load(filename_or_df, cols, **kwargs)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df

        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)
        
        self.bias = self._set_bias(H)
        self.cols_size = self._set_cols_size(H)
        self.maxs = self.data.max().values
        self.mins = self.data.min().values
        
        super(CsvTable, self).__init__(name, self.columns, pg_name)
        
    def _load(self, filename, cols, **kwargs):

        df = pd.read_csv(filename, usecols=cols, **kwargs)
        if cols is not None:
            df = df[cols]

        return df

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """

        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        if cols is None:
            cols = data.columns
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)
        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            col.SetDistribution(data[c].value_counts(dropna=False).index.values)
            columns.append(col)
            
        return columns
    
    def _set_bias(self, H):
        
        bias = [0.] * len(self.columns)
        if H.noise_value is not None:
            for i, ss in enumerate(H.noise_value.split(',')):
                bias[i] = float(ss)
                
        return bias
    
    def _set_cols_size(self, H):
        
        sizes = [1] * len(self.columns)
        
        if H.discrete:
            for i, encoding_mode in enumerate(H.encoding_modes):
            
                n = self.columns[i].distribution_size
                
                if encoding_mode == 'binary':    
                    sizes[i] = max(1, int(np.ceil(np.log2(n))))
                        
                elif encoding_mode == 'onehot':
                    sizes[i] = n
    
    
class TableDataset(torch.utils.data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, table):
        super(TableDataset, self).__init__()
        self.table = copy.deepcopy(table)

        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.tuples = torch.as_tensor(
            self.tuples_np.astype(np.float32, copy=False))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]


def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids