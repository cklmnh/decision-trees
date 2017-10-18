'''
Created on Oct 15, 2017

@author: surya
'''

class Node(object):
    '''
    Represents a generic node in the decision tree.
    
    Attributes:
        name: Represents the node's attribute name.
        children: List of child nodes.
        targetVal: Target value of the class if this is a leaf node.
        mostFrequentAttrVal: Most frequent value of the attribute based on training instances at this node.
        depth: Depth of the node in the tree. Depth of root is 0.
    '''
    
    
    def __init__(self, name, children, mostFrequentAttrVal, targetVal, parent_attr_val, depth):
        self.name = name
        self.children = children
        self.mostFrequentAttrVal = mostFrequentAttrVal
        self.targetVal = targetVal
        self.parent_attr_val = parent_attr_val
        self.depth = depth
        
    def ToString(self):
        """
        Convert attribute to string representation
        """
        delim = "  " * (self.depth + 1)
        if len(self.children) > 0:
            if self.depth == 0:
                return (self.name, ": (freqVal: %s) (depth: %d)\n"  % (self.mostFrequentAttrVal, self.depth),
                     delim, ("\n%s" % delim).join([x.ToString() for x in self.children]))
            else:
                return (self.parent_attr_val, ("\n%s" % delim), self.name, ": (freqVal: %s) (depth: %d)\n"  % 
                        (self.mostFrequentAttrVal, self.depth),
                     delim, ("\n%s" % delim).join([x.ToString() for x in self.children]))
        else:
            return self.parent_attr_val, ("\n%s" % delim), self.name, " = ", self.targetVal
        