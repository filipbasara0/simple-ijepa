# Simple I-JEPA

A simple and efficient PyTorch implementation of Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).

## Results

The model was pre-trained on 100,000 unlabeled images from the `STL-10` dataset. For evaluation, I trained and tested logistic regression on frozen features obtained from 5k train images and evaluated on 8k test images, also from the `STL-10` dataset.

Linear probing was used for evaluating on features extracted from encoders using the scikit LogisticRegression model. Image resolution was `96x96`.

More detailed evaluation steps and results for [STL10](https://github.com/filipbasara0/simple-ijepa/blob/main/notebooks/linear-probing-stl.ipynb) can be found in the notebooks directory. 

| Dataset       | Approach    | Encoder           | Emb. dim | Patch size| Num. targets | Batch size | Epochs   | Top 1% |
|---------------|-------------|-------------------|----------|-----------|--------------|------------|----------|--------|
| STL10         | I-JEPA      | VisionTransformer | 512      | 8         | 4            | 256        | 100      | 77.07  |

All experiments were done using a very small and shallow VisionTransformer (only 11M params) with following parameters:
* embbeding dimension - `512`
* depth (number of transformers layers) - `6`
* number of heads - `6`
* mlp dim - `2 * embedding dimension`
* patch size - `8`
* number of targets - `4`

The mask generator is inspired by the original paper, but sligthly simplified.

## Usage

### Instalation

To setup the code, clone the repository, optionally create a venv and install requirements:

1. `git clone git@github.com:filipbasara0/simple-ijepa.git`
2. create virtual environment: `virtualenv -p python3.10 env`
3. activate virtual environment: `source env/bin/activate`
4. install requirements: `pip install .`


### Examples

`STL-10` model was trained with this command:

`python run_training.py --fp16_precision --log_every_n_steps 200 --num_epochs 100 --batch_size 256`

### Detailed options
Once the code is setup, run the following command with optinos listed below:
`python run_training.py [args...]⬇️`

```
I-JEPA

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path where datasets will be saved
  --dataset_name {stl10}
                        Dataset name
  -save_model_dir SAVE_MODEL_DIR
                        Path where models
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  --fp16_precision      Whether to use 16-bit precision for GPU training
  --emb_dim EMB_DIM     Transofmer embedding dimm
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log every n steps
  --gamma GAMMA         Initial EMA coefficient
  --update_gamma_after_step UPDATE_GAMMA_AFTER_STEP
                        Update EMA gamma after this step
  --update_gamma_every_n_steps UPDATE_GAMMA_EVERY_N_STEPS
                        Update EMA gamma after this many steps
  --ckpt_path CKPT_PATH
                        Specify path to ijepa_model.pth to resume training
```

## Citation

```
@misc{assran2023selfsupervisedlearningimagesjointembedding,
      title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture}, 
      author={Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
      year={2023},
      eprint={2301.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.08243}, 
}
```
