"""
Main script
"""
import sys
import time
import traceback
import run_module
import experiment_module

print(sys.argv)
BASE_PATH = '/projects/p32342'
experiment_name = sys.argv[1]
print(experiment_name)
experiment_save_file = f"{BASE_PATH}/output/{experiment_name}/{experiment_name}.pkl"
experiment = experiment_module.Experiment.load(experiment_name)

if sys.argv[2] == 'submit':
    experiment.run_experiment()
elif sys.argv[2] == 'gather':
    experiment.gather_results()
elif sys.argv[2] == 'process':
    experiment.process_experiment()
    minutes_to_sleep = 20
    time.sleep(minutes_to_sleep * 60)
    experiment.gather_results()
elif sys.argv[2] == 'prototype':
    run = experiment.runs_array[0]
    successful = run.successful()
    if successful:
        print('Run successful. Preparing to process results...')
        variables_names = experiment.variables['names']
        try:
            run.process_run(variables_names)
        except Exception as e:
            print(f"An error occured: {e}")
            traceback.print_exc(limit=10)
    else:
        run.submit_run()
else:
    run_name = sys.argv[2]
    print(run_name)
    values = run_module.Run.get_values_from_name(run_name)
    run = run_module.Run(experiment, values)
    print(run)
    if sys.argv[3] == 'submit':
        run.submit_run()
    elif sys.argv[3] == 'process':
        run.process_run(experiment.variables['names'])
    elif sys.argv[3] == 'prototype':
        run.process_run(experiment.variables['names'])
        results = run.results
        output_path = BASE_PATH + '/code/results.txt'
        with open(output_path, 'w') as file:
            file.write(str(results))
    else:
        print("Incorrect usage.")
