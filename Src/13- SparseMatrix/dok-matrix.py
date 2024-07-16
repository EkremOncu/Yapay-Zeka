import numpy as np

class DokMatrix:
    def __init__(self, nrows, ncols):
        self._nrows = nrows
        self._ncols = ncols
        self._dok = {}
        
    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError('index must have two dimension')
        if index[0] < 0 or index[0] >= self._nrows:
            raise IndexError('index out of range')
        if index[1] < 0 or index[1] >= self._ncols:
            raise IndexError('index out of range')
        
        return self._dok.get(index, 0)
    
    def __setitem__(self, index, val):
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError('index must have twho dimension')
        if index[0] < 0 or index[0] >= self._nrows:
            raise IndexError('index out of range')
        if index[1] < 0 or index[1] >= self._ncols:
            raise IndexError('index out of range')
        
        self._dok[index] = val
        
    @property
    def shape(self):
        return self._nrows, self._ncols
    
    @property
    def size(self):
        return self._nrows * self._ncols
    
    def __len__(self):
        return self._nrows
    
    @staticmethod
    def array(a):
        if not isinstance(a, list):
            raise TypeError('argument must be Python list')
        nrows = len(a)
        ncols = len(a[0])
        for i in range(1, nrows):
            if len(a[i]) != ncols:
                raise ValueError('matrix rows must have the same number of elements')
        dm = DokMatrix(nrows, ncols)
        for i in range(nrows):
            for k in range(ncols):
                if a[i][k] != 0:
                    dm._dok[(i, k)] = a[i][k]
        return dm
    
    def todense(self):
        array = np.zeros((self._nrows, self._ncols))
        for index, val in self._dok.items():
            array[index] = val
        return array
    
    def __str__(self):
        smatrix = ''
        for i in range(self._nrows):
            sline = ''
            for k in range(self._ncols):
                if sline != '':
                    sline += ' '
                sline += str(self._dok.get((i, k), 0))
            if smatrix != '':
                smatrix += '\n'
            smatrix += sline
            
        return smatrix
    
    def __repr__(self):
        smatrix = ''
        for index, val in self._dok.items():
            if smatrix != '':
                smatrix += '\n'
            smatrix += f'({index[0]}, {index[1]}) ---> {val}'
            
        return smatrix
  
        
a = [[1, 0, 3], [0, 0, 1], [5, 0, 0]]
dok = DokMatrix.array(a)

print(dok)
print('-' * 10)

dok[1, 1] = 8
print(dok)
print('-' * 10)

print(repr(dok))









    
    
        
    
    
            
            
             