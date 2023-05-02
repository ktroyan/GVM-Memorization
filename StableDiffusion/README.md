# StableDiffusion on a Juptyer Notebook       

The following guide is based on the tutorial [Jupyter on Euler](https://gitlab.ethz.ch/sanagnos/Jupyter-on-Euler-or-Leonhard-Open/-/tree/master/). If more details are needed, please refer to the original tutorial.


## Installation
 

1. Run the script `commands.sh`. It will load necessary modules, and install the required packages if they are not installed yet.  

Here `$USERNAME` is your username on Euler server.
```
$ USERNAME=your_username
$ chmode 755 commands.sh
$ ./commands.sh $USERNAME
```


2. If you need to send a file to Euler, use the following command:

Here `$USERNAME` is your username on Euler, and `$DIRECTORY` is the directory where you want to send the file.

In step 1, you have created a directory `generative_models` in the home directory. So, we will send the file to this directory.

```
$ USERNAME=your_username
$ DIRECTORY=generative_models
$ scp sd_diffusers.ipynb $USERNAME@euler:/cluster/home/$USERNAME/$DIRECTORY
```


3. In `jnb_config` change necessary variables. In particular, you need to change only `JNB_USERNAME` variable.

4. Make script `start_jupyter_nb.sh` executable and run it using the configuration file `jnb_config`. 

```
$ chmod 755 start_jupyter_nb.sh
$ ./start_jupyter_nb.sh -c jnb_config
```

It will output something like this:
```
Waiting for jupyter notebook to start, sleep for 10 sec
```
That means that the script is waiting for the jupyter notebook to start. It will take some time, so be patient. After the jupyter notebook starts, it will open jupyter notebook in a local browser.