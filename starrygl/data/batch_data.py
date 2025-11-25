class BatchData:
    '''
 
    Args:
        cache_index: int, the index of the data in the cache
        attr: dict, the attributes of the data
    '''
 
    def __init__(self, cache_index, attr):
        self.cache_index = cache_index
        self.attr = attr    

    def request_next_batch(self):
        pass
        
class TemporalBatchData(BatchData):
    def __init__(self):
        super().__init__()
        
    def request_next_batch(self):
        return super().request_next_batch()
    
class ChunkBatchData(BatchData):
    '''
 
    Args:
        cache_index: int, the index of the data in the cache
        attr: dict, the attributes of the data
    '''
 
    def __init__()
    