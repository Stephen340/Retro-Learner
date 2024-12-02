# Retro-Learner

In order to get this repository working, the ZIP folder associated with this Final Report turn-in will be required.

Once the ZIP folder is obtained, extract the contents. There should be 2 ROM folders, and 8 '~.pth' models.

The two ROM folders are already set up. Changing these may negatively impact the behavior in this repository.

## Setup

First, note that Python 8 or lower is required for Gym-Retro compatibility. We recommend Python 8 (as that is the version this repo was tested with).

Install the requirements.txt file for packages. For most systems, the command may look like:

pip install -r requirements.txt

Modify this command for your environment if necessary if it takes a different command.

Some requirements here are finicky, and may require others in order to install themselves. We have worked on the order of package installation, but in the case that
for some reason there are issues with the initial pip installation on your machine, we recommend retrying, or manually installing the packages that failed. 

NOTE: While all models were trained using GPU compatibility with TorchVision (and thus the model train files may not be runnable unless this is personally added),
we have modified this repository to instead work only with the basic torch import and CPU processing for the DEMO files.

Once requirements are installed, the ROMS will need to be imported into Retro. Bring the ROM folders into the repository workspace.

E.g., "C:\Users\usr\PythonProjects\Retro-Learner\~workspace"

Once both ROM folders are in the workspace, run the following commands to import them into the repository but MAKE SURE to update the paths to your system's path to the Retro-Learner workspace:

python -m retro.import 'C:\Users\usr\PythonProjects\Retro-Learner\SonicRom'

python -m retro.import 'C:\Users\usr\PythonProjects\Retro-Learner\SMBRom'

With this, both ROMs and their environments, levels, states, and associated data should be imported into the repo.

## Running a Demo

Running a demo is simple.

Bring all 8 .pth models into the Retro-Learner workspace. There are four demo files:

### Demo Files
#### Mario_Demo.py

#### Sonic_Demo.py

#### MtS_Demo.py

#### StM_Demo.py


Each Demo is preprogrammed to run the finetuned version of the associated model. This can be manually updated in the python file to the pretrained version if desired.

Like with importing the ROMs, you will also need to manually update the path to the .pth models in the python files so that your system's path is recognized. If the DEMO file is not
updated to recognize your system's path to the .pth file, the DEMOs will not work.

Once the path is updated appropiately, simply run the associated file. This command may look like 'python ./MarioDemo.py' depending on your environment.
If paths are updated properly, this should begin the demo. The model cannot be interacted with, as the agent will play the game by itself.
