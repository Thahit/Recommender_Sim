# Exploring the Modelling of User Arrival Times of Online Streaming Sites
We aim to generate synthetic user data that mimics real-world interactions to experiment with recommender systems.\\
   Given the last interaction time t<sub>i</sub>and interactions r<sub>i</sub> generate predictions for the next time the user visits the site/app.  t <sub>i+1~ f<sub>θ</sub>(|t<sub>i</sub>, r<sub>i</sub>)$. Our idea is that users have a hidden state that evolves over time and dictates when users interact with the system.

## Data processing
After following the instructions below (Accordion: a Trainable Simulator for Long-Term Interactive Systems), use the notebook notebooks/preprocess_for_new.ipynb to continue processing the data.  

## Models
For a nicer explanation of the models, I refer to the poster.

### Model as Partially Deterministic Markov Process (PDMP)

### Alternative Model 1: Function Approximation

### Alternative Model 2: Density Estimation

## Accordion: a Trainable Simulator for Long-Term Interactive Systems
The starting point of this repo is the https://github.com/jamesmcinerney/accordion repository.
The following text is the content of its README file.

***This is prototype code for research purposes only.***

The code implements a trainable Poisson process simulator for interactive systems. The purpose of this repository is to provide the implementation used in the public experiments described in the RecSys '21 <a href="https://dl.acm.org/doi/abs/10.1145/3460231.3474259">publication</a>. See that paper for more details on the modeling and inference steps.

There are two modules:
+ `simtrain` -> contains core logic for building the simulator and training on observed data
+ `experiment` -> enables experiments using a trained simulator

There are three notebooks:
+ `notebooks/process_contentwise_data.ipynb` -> notebook to process the <a href="https://github.com/ContentWise/contentwise-impressions">ContentWise dataset</a> into a format that can be used to train the simulator
+ `notebooks/train_simulator.ipynb` -> notebook demonstrating how the simulator is trained
+ `notebooks/boltzmann_study.ipynb` -> notebook to make a Boltzmann exploration hyperparameter sweep using simulation


#### Setup

1. Download the <a href="https://github.com/ContentWise/contentwise-impressions">ContentWise dataset</a> please use the CSV data from their folder `ContentWiseImpressions/data/ContentWiseImpressions/CW10M-CSV/`
2. Set your base paths in the file `notebooks/paths.py` to point to where you downloaded the data and where intermediary and processed files will live
3. Adjust the `simtrain.SETTINGS.py` file according to the available computational resources (i.e. `N_SUBSAMPLE_USERS`, `NUMEXPR_MAX_THREADS`). Please be aware, the initial loading and first step processing of the data in `process_contentwise_data.ipynb` takes a lot of time and memory (~4 hours on large single instance). Beyond this point, user subsampling results in a smaller system of users and items that helps speed up the code for the proof of concept.


#### Requirements

Requires the following packages:

`numpy>=1.18.2`

`pandas>=0.25.3`

`matplotlib>=3.2.1`

`scipy>=1.4.1`

`scikit-learn>=0.22.2`

`tensorflow>=1.15.0`

`grpcio>=1.24.3`
                      
`tqdm>4.62.2`


#### Bibtex paper reference

Cite as:

`@inproceedings{mcinerney2021accordion,
 title={Accordion: A Trainable Simulator for Long-Term Interactive Systems},
 author={McInerney, James and Elahi, Ehtsham and Basilico, Justin and Raimond, Yves and Jebara, Tony},
 booktitle={Fifteenth ACM Conference on Recommender Systems},
 pages={102--113},
 year={2021}
}`
