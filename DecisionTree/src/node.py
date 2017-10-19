'''
Created on Oct 15, 2017

@author: surya
'''

class Node(object):
    '''
    Represents a generic node in the decision tree.
    
    Attributes:
        attr: Represents the attribute that forms the node.
        children: List of child nodes.
        targetVal: Target value of the class if this is a leaf node.
        mostFrequentAttrVal: Most frequent value of the attribute based on training instances at this node.
        depth: Depth of the node in the tree. Depth of root is 0.
        parent_attr_val: Indicating the value of its parent attribute that created this node.
        isLeaf: Indicates if it's a leaf node.
    '''
    
    
    def __init__(self, attr, children, mostFrequentAttrVal, targetVal, parent_attr_val, depth):
        self.attr = attr
        self.children = children
        self.mostFrequentAttrVal = mostFrequentAttrVal
        self.targetVal = targetVal
        self.parent_attr_val = parent_attr_val
        self.depth = depth
        self.isLeaf = self.targetVal == self.mostFrequentAttrVal
        
    def ToString(self):
        """
        Convert attribute to string representation
        """
        delim = "  " * (self.depth + 1)
        if self.isLeaf:
            return self.parent_attr_val + ("\n%s" % delim) + "%s = %s" % (self.attr.name, self.targetVal)
        else:
            if self.depth == 0:
                return (self.attr.name + " (%d):\n"  % self.depth + delim + ("\n%s" % delim).join([x.ToString() for x in self.children]))
            else:
                return (self.parent_attr_val + ("\n%s" % delim) + self.attr.name + " (%d):\n"  % 
                        self.depth + delim + " " + ("\n%s " % delim).join([x.ToString() for x in self.children]))
            
        