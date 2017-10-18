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
    
    def preprocess(self):
        for target_attr_val in self.target_attr.values:
            train_subset = [instance for instance in self.instances if instance[self.target_attr.idx] == target_attr_val]
            
            missing_attr_val = {}
            for attr in self.attributes:
                all_values = [instance[attr.idx] for instance in train_subset]
                missing_attr_val[attr.idx] = majority_attr_val(all_values)
            
            for instance in train_subset:
                for idx, attr_val in enumerate(instance):
                    if attr_val is None:
                        instance[idx] = missing_attr_val[idx]
        
    def printInstances(self):
        for instance in self.instances:
            print ', '.join(map(str, instance[:]))
        
        print "\n\n"
            
def majority_attr_val(values):
    freq = {}
    highest_freq = 0
    most_freq_val = values[0]
    for val in values:
        if val is not None:
            if freq.has_key(val):
                freq[val] += 1
            else:
                freq[val] = 1
            
            if freq[val] > highest_freq:
                highest_freq = freq[val]
                most_freq_val = val

    return most_freq_val