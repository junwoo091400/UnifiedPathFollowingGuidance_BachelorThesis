<img align="right" height="60" src="https://user-images.githubusercontent.com/5248102/126074528-004a32b9-7911-486a-9e79-8b78e6e66fdc.png">

# windywings-gym

WindyWings Gym is an [OpenAI Gym](https://www.gymlibrary.dev/content/basic_usage/) environment for fixed wing platforms.

## Setup

Install the Gym environment and dependencies

```bash
pip install -r requirements.txt
```

### Windows setup notes with Python 3.11

In Windows, the above setup script with Python 3.11 will likely fail, because `torch` module isn't available / `gym` module has dependencies with installation errors.

> Note: I'm not sure if the `torch` module missing issue would arise in Ubuntu dev environment as well, but the `pygame` build issue will probably occur if using Python 3.11.

[pygame](https://github.com/pygame/pygame) is needed by the gym module ([related issue](https://github.com/openai/gym/issues/2691)) for rendering the simulation. However, as there is no Wheel binary release ([related article](https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/)) for Pygame, MSVC compiler needs to be used to generate the .pyd files.

1. Comment out `torch` line in the `requirements.txt`, to avoid installation.
2. Download MSVC v140 compiler, as documented [here](https://www.pygame.org/wiki/CompileWindows#Step%201,%20Get%20a%20C/C++%20compiler.)
3. Download pre-release version via: `pip install pygame --pre` (reason explained [here](https://github.com/pygame/pygame/issues/3522#issuecomment-1293981862)), to avoid compilation issue due to recent upgrade of Python 3.11, where the header `longintrepr.h` was moved.

## Running the simulation

In order to run the simulation, simply execute the file fw_altitude_control.py. All relevant parameters can be set directly in this file, including the simulations that should be carried out and the corresponding logfile paths.

```bash
python3 simulations/fw_altitude_control.py
```
### Lateral kinematics model
A lateral kinematics of a fixed wing model can be run by the following example
```bash
python3 examples/fw_lateral_control.py
```

## Testing

To test the installation, run the following script

```bash
python3 simulations/test.py
```
