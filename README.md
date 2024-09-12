How to Run and Execute the Spectral Simulation Project using Python
Name:- Gurmeher Singh Puri
Matriculation Number:- 5776434

1. Getting the Project
You can either clone the project from the Git repository or use the provided ZIP file. Here are the steps for both options:

Option 1: Using the ZIP file.
1. Download the submitted ZIP file (Spectral-Simulation-Methods.zip).
2. Unzip the file on your system.
3. Navigate to the extracted project directory.
   Example: `cd /path/to/unzipped/folder/ Spectral-Simulation-Methods`

Option 2: Clone from Git
1. Open your terminal.
2. Run the following command to clone the repository:
   ```bash
   git clone https://github.com/Gurmeher23/Spectral-Simulation-Methods
   ```
3. Navigate to the project directory:
   ```bash
   cd Spectral-Simulation-Methods
   ```
2. Installing Poetry
Poetry is a dependency management tool that helps in setting up and running the project. Follow the steps below to install Poetry on your system:
For Linux/MacOS, open your terminal and run the following command:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
For Windows, follow the instructions provided on the official Poetry website: https://python-poetry.org/docs/#installation or 
if you have pipx installed directly run the command in bash ‘’’ pipx install poetry’’’
3. Installing Dependencies
Once Poetry is installed, navigate to the project directory (where `pyproject.toml` is located) and run the following command to install the dependencies:
```bash
poetry install
```
This command will install all the dependencies specified in the `pyproject.toml` file, such as NumPy, Matplotlib, and others required for the simulation.
4. Running the Project
After installing the dependencies, you can run the project using the following steps:

Running the Main Script
Once poetry is installed by following the poetry Step 2, make sure to follow step 3 by running the command ‘’ poetry install’’ in the project directory bash so that all the dependencies are installed. 
Go to the project directory cd Spectral-Simulation-Methods/
Milestone 1:
Run the command in bash/terminal to run the file of milestone1.
poetry run python Milestone1/combined_script.py

Milestone 2:
Run the command in bash/terminal to run the file of mileston2.
poetry run python Milestone2/taylor_green_vortex.py

Milestone 3:
Run the command in bash/terminal to run the file of mileston3.
poetry run python Milestone3/turbulence_simulation_03.py

Milestone 4:
Run the command in bash/terminal to run the file of mileston4.
poetry run python Milestone4/turbulence_simulation04.py

