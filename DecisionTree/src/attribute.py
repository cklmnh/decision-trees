'''
Created on Oct 15, 2017

@author: surya
'''

class Attribute(object):
    '''
    Represents an attribute in the data.
    
    Attributes:
        name: Represents the attribute name.
        idx: Index of the attribute in the data instance representation.
        values: Contains the list of unique values the attribute can take.
    '''


    def __init__(self, name, idx, values):
        self.name = name
        self.idx = idx
        self.values = values
        
    def ToString(self):
        """
        Convert attribute to string representation
        """
        return self.name + ": " + ', '.join(self.values)
    