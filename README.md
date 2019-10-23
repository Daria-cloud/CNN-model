# Diagnosing Cancer from Immune Receptor Sequences 

# Description

T lymphocytes are cells of the immune system that attack and destroy virus-infected cells, tumor cells and cells from transplanted organs. This occurs because each T cell is endowed with a highly specific receptor that can bind to an antigen present at the surface of another cell. A direct detection of asymptotical early-stage cancers is usually challenging due to small tumor volume and limited amount of detectable alterations in the circulation. Here, we developed a deep learning model based on the convolutional neural network to distinguish sequences of cancer-associated T cell receptors from non-cancer ones. This work sets the foundation to utilize peripheral blood T cell receptor repertoire for non-invasive early cancer detection.

First, we took sequences of CDR3 regions of all studied T cell receptors (TCR). CDR3 regions are known to play a major role in recognition of antigens. The amino acids residues of each CDR3 region were then encoded using the BLOSUM62 matrix, which contains information about 
amino acids similarity. The TCR sequences were processed with 8 convolutional filters of size (20,2), followed by maxpooling and 16 convolutional filters of size (1,2) with maxpooling. The results were connected to a dense layer of 10 hidden neurons connected to the output neuron. 

The network was trained using 25 epochs and the model with best validation accuracy was saved (‘best_model.h5’). The model was futher validated on 10% of the data and gave AUC ROC value of 82%.

For a complete description of this approach see our publication:
to be included

# Requirements

* [Python3](https://www.python.org/)
* [Keras](https://keras.io/)
* [NumPy](http://www.numpy.org/) 
* [Scikit-learn] (https://scikit-learn.org)

# Primary Files

 * CNN_blosum.py
 * data_io_blosum.py
 * Input files are stored in folder ‘Data’

