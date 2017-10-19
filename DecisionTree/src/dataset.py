'''
Created on Oct 15, 2017

@author: surya
'''

import arff
from attribute import Attribute

class Dataset(object):
    '''
    Representation of the input data containing attributes and training instances.
    
    Attributes:
        attributes: List of Attribute objects.
        target_attr: Target attribute to predict.
        instances: List of training instances.
    '''
    def __init__(self, filepath, target_attr_name):
        input_data = arff.load(open(filepath, 'rb'))
        self.instances = input_data.get("data")
        self.attributes = []
        
        attributes_list = input_data.get("attributes", [])
        for idx, attribute_info in enumerate(attributes_list):
            if len(attribute_info) == 2:
                name = attribute_info[0]
                attribute = Attribute(name, idx, attribute_info[1])
                if name == target_attr_name:
                    self.target_attr = attribute
                else: 
                    self.attributes.append(attribute)
        
    def printInstances(self):
        for instance in self.instances:
            print ', '.join(map(str, instance[:]))
        
        print "\n\n"