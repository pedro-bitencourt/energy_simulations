"""
File name: run_module.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: this file implements the Run class and related methods.
"""

import os
import re
import shutil
from pathlib import Path
import logging
import subprocess

from src.auxiliary import try_get_file, submit_slurm_job
from .constants import initialize_paths_run, create_run_name

logger = logging.getLogger(__name__)

MEMORY_REQUESTED = '5G'


class Run:
    """
    Represents a run and implements some general purpose methods for it.

    Args:
        - folder: Path determining the folder where the run is stored.
        - general_parameters: Dictionary containing general parameters for the run.
            o xml_basefile: Path to the basefile for the .xml file.
            o name_subfolder: Name of the subfolder where the run is stored.
            o requested_time_run: Time requested for the run in the cluster.
        - variables: Dictionary containing the variables for the run. Each entry is 
        a variable, which is a dictionary containing:
            o value: Value of the variable.
            o pattern: Pattern to be substituted in the .xml file.
    Attributes:
        - name: Name of the run, determined by the folder name.
        - paths: Dictionary containing relevant paths for the run.
    """

    def __init__(self,
                 parent_folder: Path,
                 general_parameters: dict,
                 variables: dict[str, dict]):
        parent_folder: Path = Path(parent_folder)
        self.variables: dict[str, dict] = variables
        self.general_parameters: dict = general_parameters

        self.name: str = create_run_name(variables)
        self.parent_name: str = parent_folder.parts[-1]

        # Initialize relevant paths
        self.paths: dict = initialize_paths_run(
            parent_folder, self.name, self.general_parameters['name_subfolder'])

        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

    def tear_down(self):
        """
        Deletes the folder and its contents.
        """
        if self.paths['folder'].exists():
            logger.info("Deleting folder %s...", self.paths['folder'])
            try:
                shutil.rmtree(self.paths['folder'])
            except FileNotFoundError:
                logger.warning("Could not delete folder properly")
            logger.info("Deleted folder %s", self.paths['folder'])

    def _get_opt_and_sim_folders(self):
        '''
        Get the opt and sim folders from the subfolder path.
        '''
        opt_sim_paths = {'opt': None, 'sim': None}
        folder_ending = {'opt': "*-OPT", 'sim': "*-SIM"}
        test_files = {'opt': r'tiempos*', 'sim': r'resumIng*'}

        for key in opt_sim_paths:
            candidate_folder_list = list(
                self.paths['subfolder'].glob(folder_ending[key]))
            if candidate_folder_list:
                for folder in candidate_folder_list:
                    correct_folder = try_get_file(
                        folder, test_files[key])
                    if correct_folder:
                        opt_sim_paths[key] = folder
                        continue
        return opt_sim_paths

    def successful(self, complete: bool = False):
        """
        Check if the run was successful by searching for a resumen* file
        in the sim folder.
        """
        # Get opt and sim folders
        self.paths.update(self._get_opt_and_sim_folders())

        # Check if SIM folder exists
        sim_folder = self.paths.get('sim', False)
        if sim_folder:
            # Check if files exist
            missing_files = self.missing_files(complete)
            if missing_files:
                logger.debug(
                    "Run %s is missing files: %s", self.name, missing_files)
                return False
        else:
            logger.debug('No sim folder found for run %s in folder %s',
                         self.name,
                         self.paths['subfolder'])
            return False
        return True

    def missing_files(self, complete: bool = False):
        """
        Check if the run is missing files.
        """
        files_to_check = [r'resumen*',
                          r'EOLO_eoloDeci/potencias*.xlt',
                          r'FOTOV_solarDeci/potencias*.xlt',
                          r'DEM_demandaPrueba/potencias*.xlt'
                          ]
        # Add hydro files if complete
        if complete:
            files_to_check.append(r'HID_salto/cota*.xlt')
            files_to_check.append(r'HID_salto/potencias*.xlt')

        missing_files: list[str] = []
        # Check if files exist
        for file_name in files_to_check:
            file_found = try_get_file(self.paths['sim'], file_name)
            if not file_found:
                missing_files.append(file_name)
                logger.debug("%s does not contain file %s",
                             self.paths['sim'], file_name)
        return missing_files

    def prototype(self):
        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

        # Create the xml file
        xml_path: Path = self._create_xml()

        # Create the bash file
        bash_path: Path = self._create_bash(xml_path)

        # Print bash path
        print("Bash path:")
        print(bash_path)
        subprocess.run(['bash', bash_path])

    def submit(self, force: bool = False):
        """
        Submits a run to the cluster.
        """
        # Break if already successful
        if self.successful() and not force:
            logger.info(f"Run {self.name} already successful, skipping.")
            return None

        logger.info(f"""Preparing to submit run {self.name}""")

        logger.warning("Warning: this will overwrite the folder %s",)
        # Tear down the folder
        self.tear_down()

        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

        # Create the xml file
        xml_path: Path = self._create_xml()

        # Create the bash file
        bash_path: Path = self._create_bash(xml_path)

        # Submit the slurm job
        job_id = submit_slurm_job(bash_path, job_name=self.name)

        # Check if the job was submitted successfully
        if job_id:
            logger.info(
                f"Successfully submitted run {self.name} with jobid {job_id}")
            return job_id
        logger.error(f"Some error occurred while submitting run {self.name}")
        return job_id

    def _create_xml(self):
        """
        Creates a .xml file from the .xml template.
        """

        xml_path: Path = self.paths['folder'] / f"{self.name}.xml"
        # Open the experiment's .xml basefile
        with open(self.general_parameters['xml_basefile'], 'r') as file:
            content = file.read()

            def escape_for_regex(path):
                return path.replace('\\', '\\\\')

            # Change output path
            content = re.sub(
                r'<rutaSalidas>.*?</rutaSalidas>',
                f'<rutaSalidas>{escape_for_regex(self.paths["folder_windows"])}</rutaSalidas>',
                content
            )

            # Substitute variables values
            content = self._substitute_variables(content)

        # Write the new settings to the .xml file path
        with open(xml_path, 'w') as file:
            file.write(content)

        return xml_path

    def _substitute_variables(self, content):
        """
        Substitutes patterns in the content with the variable values.
        """
        DEGREES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        def substitute_pattern(content, pattern, value):
            """
            Substitutes a pattern in the content with a given value.
            """
            def replace_func(match):
                if '*' in match.group():
                    # Get the captured multiplier
                    multiplier = float(match.group(2))
                    # Changed from int() to float()
                    new_value = float(value * multiplier)
                    return str(new_value)
                return str(value)

            # First try to match pattern with multiplier
            content = re.sub(
                f'({pattern})(?:\*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))', replace_func, content)

            # Then match pattern without multiplier
            content = re.sub(
                f'({pattern})<', replace_func, content)

            return content

        for variable in self.variables.values():
            for degree in DEGREES:
                pattern = variable['pattern'] + f"_{degree}"
                if variable['value'] > 0:
                    value = variable['value']**degree
                else:
                    logger.info("Inputting 0 for %s at degree %s due to value  %s",
                                variable['pattern'], degree, variable['value'])
                    value = 0
                content = substitute_pattern(content, pattern, value)
            content = substitute_pattern(
                content, variable['pattern'], variable['value'])

        return content

    def _create_bash(self, xml_path: Path):
        """
        Creates a bash file to be submitted to the cluster.
        """
        bash_path = self.paths['folder'] / f"{self.name}.sh"
        xml_path = os.path.normpath(str(xml_path))
        xml_path = xml_path.replace(os.path.sep, '\\')
        job_name = f"{self.parent_name}_{self.name}"

        requested_time: float = self.general_parameters['requested_time_run']
        hours: int = int(requested_time)
        minutes: int = int((requested_time * 60) % 60)
        seconds: int = int((requested_time * 3600) % 60)
        requested_time_run: str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if self.general_parameters.get('email', None):
            email_line = f"#SBATCH --mail-user={self.general_parameters['email']}"
        else:
            email_line = ""

        temp_folder_path = f"{self.paths['folder']}/temp"
        temp_folder_path_windows = temp_folder_path.replace('/', '\\')

        with open(bash_path, 'w') as file:
            file.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time_run}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem={MEMORY_REQUESTED}
#SBATCH --job-name={job_name}
#SBATCH --output={self.paths['folder']}/{self.name}.out
#SBATCH --error={self.paths['folder']}/{self.name}.err
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --exclude=qhimem[0207-0208]
{email_line}

echo "Starting {self.name} at: $(date +'%H:%M:%S')"
export WINEPREFIX=/projects/p32342/software/.wine
mkdir -p {temp_folder_path}
module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
sleep $((RANDOM%60 + 10))
wine "Z:\\projects\\p32342\\software\\Java\\jdk-11.0.22+7\\bin\\java.exe" -Djava.io.tmpdir="Z:\{temp_folder_path_windows}" -Xmx{MEMORY_REQUESTED} -jar MOP_Mingo.JAR "Z:{xml_path}"
''')
        return bash_path
