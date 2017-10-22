'''
Created on Oct 15, 2017

@author: surya
'''

class Node(object):
    '''
    Represents a generic node in the decision tree.
    
    Attributes:
        attr: Represents the attribute that forms the node.
        targetVal: Target value of the class if this is a leaf node.
        mostFrequentAttrVal: Most frequent value of the attribute based on training instances at this node.
        parent_attr_val: Indicating the value of its parent attribute that created this node.
        depth: Depth of the node in the tree. Depth of root is 0.
    '''
    
    
    def __init__(self, attr, most_freq_attr_val, target_val, parent_attr_val, depth):
        self.attr = attr
        self.most_freq_attr_val = most_freq_attr_val
        self.target_val = target_val
        self.parent_attr_val = parent_attr_val
        self.depth = depth
        self.children = []
        
    def ToString(self):
        """
        Convert attribute to string representation
        """
        delim = "  " * (self.depth + 1)
        if len(self.children) == 0:
            return self.parent_attr_val + ("\n%s" % delim) + "%s (%d) = %s" % (self.attr.name, self.depth, self.target_val)
        else:
            if self.depth == 0:
                return (self.attr.name + " (%d):\n"  % self.depth + delim + ("\n%s" % delim).join([x.ToString() for x in self.children]))
            else:
                return (self.parent_attr_val + ("\n%s" % delim) + self.attr.name + " (%d):\n"  % 
                        self.depth + delim + " " + ("\n%s " % delim).join([x.ToString() for x in self.children]))
            
        