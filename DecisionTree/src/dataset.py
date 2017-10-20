'''
Created on Oct 15, 2017

@author: surya
'''

import arff
import random
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
        
        self.handleClassImbalance()
        
    def printInstances(self):
        for instance in self.instances:
            print ', '.join(map(str, instance[:]))
        
        print "\n\n"

    def handleClassImbalance(self):
        class_dict = {}
        for instance in self.instances:
            target_val = instance[self.target_attr.idx]
            if target_val in class_dict:
                class_dict[target_val].append(instance)
            else:
                class_dict[target_val] = [instance]
        
        class_percentages = {}
        for target_val in self.target_attr.values:
            if target_val in class_dict:
                class_percentages[target_val] = len(class_dict[target_val])/float(len(self.instances))
            else:
                class_percentages[target_val] = 0.0     
         
        self.instances = []
        self.instances.extend(class_dict[self.target_attr.values[0]])  
        neg_samples = class_dict[self.target_attr.values[1]]
        self.instances.extend(random.sample(neg_samples, len(neg_samples)/2))
        
        print len(self.instances)