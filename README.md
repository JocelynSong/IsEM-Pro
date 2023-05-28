<h1>IsEM-Pro</h1>


<h2>Installation</h2></br>
The dependencies can be set up using the following commands:</br>
```ruby
conda create -n isem-pro python=3.8 -y </br>
conda activate isem-pro </br>
conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y </br>
conda install numpy=1.19 pandas=1.3 -y </br>
```

Clone this repository by the following commands:
```ruby
git clone git@github.com:JocelynSong/IsEM-Pro.git
cd IsEM-Pro
```

<h2>Usage</h2>
Training:
First, train VAE model:
```ruby
bash train_vae_mrf.sh data_path outout_path
```
Then train the latent generative model using MCEM:
```ruby
bash train_is_vae.sh data_path outout_path first_stage_path
```
The sample number (--max-iteration-sample) is 10% of the original training data size.


Inference:
```ruby
bash generate_vae.sh data_path outout_path generation_path
```