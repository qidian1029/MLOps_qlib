import nni
from nni.experiment import Experiment
from nni.experiment.config import ExperimentConfig

def main():
    config = ExperimentConfig(
        experiment_name="SVR_tuning",
        trial_command="python nni_trial.py",
        trial_concurrency=1,
        trial_code_directory="./",
        max_trial_number=200,
        max_experiment_duration="2h",
        search_space_file="nni_search_space.json",
        tuner="TPE",
        tuner_kwargs={"optimize_mode": "minimize"},
        use_annotation=False,
        training_service="local"
    )

    experiment = Experiment(config)
    experiment.start()

if __name__ == "__main__":
    main()