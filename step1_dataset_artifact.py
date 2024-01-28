from clearml import Task, StorageManager, Dataset

project_name = "ihk-mlops"
dataset_name = "iris_dataset"

# create an dataset experiment
task = Task.init(project_name=project_name, task_name="step_1")

# only create the task, we will actually execute it later
task.execute_remotely()

# download the data
local_iris_pkl = StorageManager.get_local_copy(
    remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl')

dataset = Dataset.create(dataset_name=dataset_name, dataset_project=project_name, output_uri="s3://192.168.231.166:9000/clearml-outputs")
dataset.add_files(local_iris_pkl)
dataset.upload()

# add and upload local file containing our toy dataset
#task.upload_artifact('dataset', artifact_object=local_iris_pkl)

print("finalize dataset")
dataset.finalize()