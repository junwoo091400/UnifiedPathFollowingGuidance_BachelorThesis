<img align="right" height="60" src="https://user-images.githubusercontent.com/5248102/126074528-004a32b9-7911-486a-9e79-8b78e6e66fdc.png">

# windywings-gym

WindyWings Gym is an [OpenAI Gym](https://www.gymlibrary.dev/content/basic_usage/) environment for fixed wing platforms.

## Setup

Install the Gym environment and depdnencies

```bash
pip install -r requirements.txt
```

## Running the simulation

In order to run the simulation, simply execute the file fw_altitude_control.py. All relevant parameters can be set directly in this file, including the simulations that should be carried out and the corresponding logfile paths.

```bash
python3 simulations/fw_altitude_control.py
```

## Testing

To test the installation, run the following script

```bash
python3 simulations/test.py
```
