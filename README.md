# decision-trees
KDD Cup 2000 Problem 1: Clickstream Mining with Decision Trees

# Instructions to run:
From terminal, go to 'src' folder which contains the main.py file.

Run 'python main.py -h' to get help with usage. Below is the output of the command.

usage: main.py [-h] [--train TRAIN] [--test TEST] [--out OUT] [--conf CONF]
               [--mh MH] [--null NULL] [--cb CB] [--target TARGET]

optional arguments:
  -h, --help       show this help message and exit
  --train TRAIN    Training set file path. Default: '../training_subsetD.arff'
  --test TEST      Test set file path. Default: '../testingD.arff'
  --out OUT        Output folder name. Default: Output
  --conf CONF      List of confidence values to run. Default:
                   0,0.5,0.8,0.95,0.99
  --mh MH          Missing values handling. 0 (Default) = Option 1 in class. 1
                   = Option 2 in class.
  --null NULL      Handle null values as missing values. 0 (Default) = False.
                   1 = True.
  --cb CB          Do class balancing by under sampling. 0 (Default) = False.
                   1 = True.
  --target TARGET  Target attribute name. Default = 'Class'

# Example
For the given assignment dataset and test set, below is an example of how to run the program.

python main.py --train '../../training_subsetD.arff' --test '../../testingD.arff' --out 'Output' --conf 0.95,0.99 --mh 0 --null 0 --cb 1

# NOTE: Please note that all the parameters are optional. If you satisfy the default values for each parameter, the program can be executed by just running 'python main.py'
