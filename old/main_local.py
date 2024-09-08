"""
Main script
"""
import run_module
import experiment_module

experiment_save_file = "/Users/pedrobitencourt/quest/data/renewables/hydros_to_thermal_weekly/hydros_to_thermal_weekly.pkl" 
run_name = "2000_250"


experiment = experiment_module.Experiment.load_from_filepath(experiment_save_file)
run_output_path = "/Users/pedrobitencourt/quest/data/renewables/hydros_to_thermal_weekly/2000_250" 
values = run_module.Run.get_values_from_name(run_name)
run = run_module.Run(experiment, values, run_output_path)
run.process_run(experiment.variables['names'])
