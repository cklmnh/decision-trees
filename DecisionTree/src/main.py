'''
Created on Oct 15, 2017

@author: surya
'''

from dataset import Dataset
from tree import Tree

def main():
    sample_data_filepath = '../weather_data.arff'
    training_data_filepath = '../training_subsetD.arff'
    
    sample_target_attr_name = 'play'
    target_attr_name = 'Class'
    
    #dataset = Dataset(sample_data_filepath, sample_target_attr_name)
    
    print "Reading input data..."
    dataset = Dataset(training_data_filepath, target_attr_name)
    
    print "Successfully parsed training data.\nPreprocessing training data..."
    dataset.preprocess()
    
    print "Preprocessing done. Handled missing values."
    dtree = Tree()
    
    print "Learning tree by fitting data..."
    dtree.learn(dataset, 0.0)
    
    print "Tree building successful."
    
if __name__ == "__main__":
    main()
