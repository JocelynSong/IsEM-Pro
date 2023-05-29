<h1>IsEM-Pro</h1>


<h2>Installation</h2>
The dependencies can be set up using the following commands:

```ruby
conda create -n isem-pro python=3.8 -y 
conda activate isem-pro 
conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y 
conda install numpy=1.19 pandas=1.3 -y 
```

Clone this repository by the following commands:
```ruby
git clone git@github.com:JocelynSong/IsEM-Pro.git
cd IsEM-Pro
```

<h2>Usage</h2>
<h3>Preprocess</h3>
Training MRFs (taking avGFP as an example):

```ruby
mkdir avGFP
bash run_mrf.sh avGFP
```

<h3>Training</h3>
First, train VAE model:

```ruby
bash train_vae_mrf.sh data_path avGFP outout_path
```

Then train the latent generative model using MCEM:

```ruby
bash train_is_vae.sh data_path avGFP outout_path first_stage_path
```
The sample number (--max-iteration-sample) is 10% of the original training data size.


<h3>Inference</h3>

```ruby
bash generate_vae.sh data_path avGFP outout_path generation_path
```

generation_path/protein.txt is the final output file