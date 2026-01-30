# Disentangling multispecific antibody function with graph neural networks

We present a generative method for creating large-scale, realistic synthetic functional landscapes that capture nonlinear interactions where biological activity depends on domain connectivity. Second, we have several baselines including a graph neural network architecture and a sequence-only MLP using this data generation process. 



### Running Scaling Analysis

To reproduce the results on model performance scaling analysis (Figure 2.).  
```
python run_sweep.py model

```

## Running Transfer Learning Experiment

To run our trispecific example where we explore the effect of first pre-training on a node-level task (Figure 3.)
```
python run_sweep.py transfer
```

Note that our experiments in the paper use the full paired OAS to sample sequences, whereas here we have provided a subset of sequences in sequence_examples.csv.


## Citation

If you find this work useful, please cite our ArXiv paper:
```bibtex
@article{}, 
}
```