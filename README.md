# Active Inference for collaborative pHRI scenarios
In this repository I have implemented the planner detailed in [An Active Inference approach for intention recognition and role arbitration during assistive physical Human-Robot Interaction](https://hal.science/hal-05288256/document). The planner generates setpoints for a collaborative arm and is wrapped in a ROS 1 (Noetic) node. It uses a custom variable impedance controller implemented in [`my_franka_controllers`](https://github.com/singhbal-baljinder/my_franka_controllers). 

# :clipboard: Installation
- Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Create an environment using the `environment.yml`:
  ```
  conda env create -f <path-to-repo>/environment.yml
  ```
- Use [catkin](https://wiki.ros.org/catkin/conceptual_overview#Overview) to install the package in the [ROS workspace](https://wiki.ros.org/catkin/Tutorials/create_a_workspace):
  ```
  cd <ros-noetic-ws>
  catkin_make install
  ```

# :bicyclist: Usage
The package can be used together with [`my_franka_controllers`](https://github.com/singhbal-baljinder/my_franka_controllers) with a Franka arm:
```
roslaunch active_inference_hri experiment_real_robot.launch controller:=my_cartesian_impedance_controller robot_ip:=<your-robot-ip>
```

# Paper data and plots
The rosbags of the experiments used in the paper can be found in the data folder. It is possible to generate the plots of the paper using the `plot_experiments_data.ipynb` Jupyter notebook.

# 🖋️ Cite

```
@inproceedings{singhbal,
  TITLE = {{An Active Inference approach for intention recognition and role arbitration during assistive physical Human-Robot Interaction}},
  AUTHOR = {Singh Bal, Baljinder and Pitti, Alexandre and Cohen, Laura and Laschi, Cecilia and Ca{\~n}amero, Lola},
  URL = {https://hal.science/hal-05288256},
  BOOKTITLE = {{INTELLIGENT AUTONOMOUS SYSTEMS - 19}},
  ADDRESS = {G{\^e}nes, Italy},
  YEAR = {2025},
  MONTH = Jun,
  KEYWORDS = {Intention recognition ; Active perception ; Assistive Robotics ; Active Inference ; pHRI ; pHRI Active Inference Assistive Robotics Active perception Intention recognition},
  PDF = {https://hal.science/hal-05288256v1/file/IAS_19-11.pdf},
  HAL_ID = {hal-05288256},
  HAL_VERSION = {v1},
}

```
