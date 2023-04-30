# StableDiffusion on a Juptyer Notebook       

The following guide is based on the tutorial [Jupyter on Euler](https://gitlab.ethz.ch/sanagnos/Jupyter-on-Euler-or-Leonhard-Open/-/tree/master/). If more details are needed, please refer to the original tutorial.


## Installation
 

1. In `start_jupyter_nb.sh` change following variables or pass them as arguments to the script:

`JNB_USERNAME` - your username on Euler (line 74)  
`JNB_NUM_CPU` - number of CPUs you want to use (line 77). Set it to 4.  
`JNB_RUN_TIME` - runtime in hours (line 80).  
`JNB_MEM_PER_CPU_CORE` - memory per CPU core in MB (line 83). Set it to 4096.  
`JNB_NUM_GPU` - number of GPUs you want to use (line 86). Set it to 1.  
`JNB_WAITING_INTERVAL` - waiting interval in seconds (line 89). Set it to 10.  
`JNB_SSH_KEY_PATH` - path to your ssh key (line 92).  
`JNB_GPU_MEM` - memory per GPU in MB (line 121). Set it to "20g".

- Aferwards, change its permissions to make it executable
```
$ chmod 755 start_jupyter_nb.sh
```

2. Change the `USERNAME` variable in `commands.sh`. Make the script executable and run it. It will load necessary modules, and send `sd_diffusers.ipynb` to the Euler.
```
$ chmode 755 commands.sh
$ ./commands.sh
```

3. Run the script. 
```
$ ./start_jupyter_nb.sh
```
It will output something like this:
```
Waiting for jupyter notebook to start, sleep for 60 sec
```
That means that the script is waiting for the jupyter notebook to start. It will take some time, so be patient. After the jupyter notebook starts, it will open jupyter notebook in a local browser.
