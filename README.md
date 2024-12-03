# Retro-Learner

In order to get this repository working, the ZIP folder associated with this Final Report turn-in will be required.

Once the ZIP folder is obtained, extract the contents. There should be 2 ROM folders, and 8 '~.pth' models.

Copy and paste all contents as-is (both ROMs and all models) into the Retro-Learner workspace.

The two ROM folders are already set up. Changing these may negatively impact the behavior in this repository.

## Executable
You will need to set your Python interpreter to Python 8 for your bash environment.

If the interpreter is selected, and all ROM folders and models are brought into the workspace, an executable can be run to
automate the rest of this process. Simply run the 'run_demo.sh' file using bash and the imports will install automatically, 
the ROMs will import automatically, and the Mario_Demo.py file will run. The installation process may take a while.

This executable has been verified on a fresh system. If it does not work for some reason, we recommend following the manual process below.

PLEASE NOTE: Due to this being the executable, only one demo file can run: the basic Mario Demo. If you want to run them
using the .sh script, we recommend removing the 'pip install' line from the run_demo.sh file, and replacing the Mario_Demo.py file with
whichever demo file you prefer. See the below 'Running a Demo' section to see demo file names.

If you do not want to run an executable, follow the below instructions for manual setup.

## Setup

First, note that Python 8 or lower is required for Gym-Retro compatibility. We recommend Python 8 (as that is the version this repo was tested with).
Ensure you set your interpreter properly.

Install the requirements.txt file for packages. We recommend a fresh install without using caches packages as cached packages can cause issues 
with the package installation process. For most systems, the command may look like:

pip install -r ./requirements.txt --no-cache-dir

or if you want to use cached packages:

pip install -r requirements.txt

Modify this command for your environment if necessary. 

PLEASE NOTE: If using an environment such as PyCharm, clicking on the "Install Requirements" suggestion when the 
requirements.txt file is open will circumvent cache issues if you run into them.

NOTE: While all models were trained using GPU compatibility with TorchVision (and thus the model training files may not be runnable unless this is personally added),
we have modified this repository to instead work only with the basic torch import and CPU processing for the DEMO files.

Once requirements are installed, the ROMS will need to be imported into Retro. Bring the ROM folders into the repository workspace if not already done.

E.g., "C:\Users\usr\PythonProjects\Retro-Learner\~workspace"

Once both ROM folders are in the workspace, run the following commands to import them into the repository:

python -m retro.import 'SonicRom'

python -m retro.import 'SMBRom'

With this, both ROMs and their environments, levels, states, and associated data should be imported into the repo.

## Running a Demo

Running a demo is simple.

Bring all 8 .pth models into the Retro-Learner workspace if not already done. There are four python demo files (MtS = Mario to Sonic, StM = Sonic to Mario) that can be run:

### Demo Files
#### Mario_Demo.py

#### Sonic_Demo.py

#### MtS_Demo.py

#### StM_Demo.py


Each Demo is preprogrammed to run the finetuned version of the associated model. This can be manually updated in the python file to the pretrained version if desired.

Once the desired version is determined, simply run the associated file. This command may look like 'python ./MarioDemo.py' depending on your environment.
If paths are updated properly, this should begin the demo. The model cannot be interacted with, as the agent will play the game by itself.
