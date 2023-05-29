# Issues

Any notable issues are documented here for future reference.

## Jetbot


### General Setup
- Hardware [Documentation](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/hardware-setup)

- Software [documentation](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/software-setup)

Hardware and software are already setup. It's important to note that the ```run_trial.ipynb``` file found in the ```jetbot/``` folder needs to be run on a specific juptyter notebook. 

You can access this by typing ```http://<jetbot_ip_address>:8888``` on your browser. The IP address can be found on the LED display on the jetbot once it is turned on. 

Using the research laptop:
 - Slack Matt for the username and password. Once you are logged in, you can create a new account and login using your own credentials.



### WiFi

The jetson nano makes connecting to IllinoisNet extremely difficult, however connecting to IllinoisNetGuest is quite simple. The problem is that the guest connection only lasts for a few days, creating a very annoying situation where you must reconnect. Reconnecting the jetbot is easiest with a monitor, keyboard, and mouse, however ideally we should only have to connect once.

#### Solution

Can connect via ```go.illinois.edu/newwifi``` instead. The JetsonNano is already connected to IllinoisNet and this should not have to be done again. If you're having trouble connecting the the jupyter notebook for the JetBot, you may have to plug the JetsonNano into a monitor and reconnect it to the network. Hopefully this doesn't happen again since it is connected to IllinoisNet instead of the guest network.

### Camera

The jetbot camera initially had a strange pink tint when taking pictures and videos. There is a fairly easy [fix](https://jonathantse.medium.com/fix-pink-tint-on-jetson-nano-wide-angle-camera-a8ce5fbd797f) if you know where to look! Run the following commands on the jetson nano:

```
wget https://www.waveshare.com/w/upload/e/eb/Camera_overrides.tar.gz
tar zxvf Camera_overrides.tar.gz
sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
```

This should download the correct software to get rid of the pink tint.
