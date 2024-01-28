import pickle
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
import joblib

project_name = "ihk-mlops"
task = Task.init(project_name=project_name, task_name="step_2")

args = {
    'dataset_name': '',
    'random_state': 42,
    'test_size': 0.2,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

if args['dataset_name']:
    dataset = Dataset.get(
        dataset_project=project_name,
        dataset_name=args['dataset_name'],

    )
    dataset_directory = dataset.get_local_copy()
    # open the local copy
    iris = pickle.load(open(dataset_directory + '/' + dataset.list_files()[0], 'rb'))

    # "process" data
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args['test_size'], random_state=args['random_state'])

    joblib.dump(X_train, "X_train.pkl", compress=True)
    joblib.dump(X_test, "X_test.pkl", compress=True)
    joblib.dump(y_train, "y_train.pkl", compress=True)
    joblib.dump(y_test, "y_test.pkl", compress=True)

    # upload processed data
    print('Uploading process dataset')
    dataset_train_test = Dataset.create(dataset_name="train_test_data", dataset_project=project_name,
                             output_uri="s3://192.168.231.166:9000/clearml-outputs")

    dataset_train_test.add_files("X_train.pkl")
    dataset_train_test.add_files("X_test.pkl")
    dataset_train_test.add_files("y_train.pkl")
    dataset_train_test.add_files("y_test.pkl")
    dataset_train_test.upload()
    dataset_train_test.finalize()

    print('Notice, files are uploaded in the background')
    print('Done')
else:
    raise ValueError("Missing dataset name!")



